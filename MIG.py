import os
import sys
import time
from builtins import enumerate

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append("..")
sys.path.append("../..")
from eval_yolo import get_yolo_boxes, get_predicted_boxes, calculate_map

IMAGE_SIZE = 640

def BCE_loss_boxes(output):
    """
    Binary Cross-Entropy Loss for objectness scores.
    """
    target = torch.zeros_like(output)
    loss = torch.nn.functional.binary_cross_entropy(output, target)
    return loss

def compute_image_gradients(model, loss_fn, input_data):
    input_tensor = input_data.clone().detach().requires_grad_(True)
    outputs = model.model(input_tensor)

    # Check if outputs is a tuple or list and extract the predictions tensor
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

    # Now outputs should be a tensor
    # outputs shape: (batch_size, num_predictions, num_features)
    # Extract objectness scores (assumed to be at index 4)
    objectness = outputs[..., 4]  # Shape: (batch_size, num_predictions)
    # objectness = outputs[:, 4,:]  # Shape: (batch_size, num_predictions)

    # Apply sigmoid if necessary (if model outputs logits)
    objectness = torch.sigmoid(objectness)

    # Compute loss
    loss = loss_fn(objectness)

    loss.backward()
    image_gradients = input_tensor.grad.clone()
    input_tensor.grad.zero_()
    model.model.zero_grad()
    return image_gradients

def get_preprocessor(model, batch_size):
    custom = {"conf": 0.25, "batch": batch_size, "save": False, "mode": "predict"}
    args = {**model.overrides, **custom}
    model_predictor = model._smart_load("predictor")(overrides=args, _callbacks=model.callbacks)
    model_predictor.setup_model(model=model.model, verbose=False)
    model_predictor.imgsz = IMAGE_SIZE
    return model_predictor.preprocess

def MIGAttack(model, img, labels, epsilon, iterations, momentum, loss, baseline, preprocessor, order_of_approximation):
    accumulated_integrated_grad = torch.zeros_like(img).cuda()
    alpha = epsilon / iterations
    original_image = img.clone()
    batch_size = img.size(0)

    for i in range(iterations):
        # Line 5 algorithm 1
        approximated_gradients = torch.zeros_like(img).cuda()
        # EQ 4 second term
        for j in range(order_of_approximation):
            scaled_imgs = baseline + (j / order_of_approximation) * (img - baseline)
            grad = compute_image_gradients(model, loss, scaled_imgs)
            approximated_gradients += grad * (1 / order_of_approximation)
        # EQ4 sum
        integrated_grad = (img - baseline) * approximated_gradients
        # Line 7 algorithm 1
        # Compute the L1 norm over each image in the batch
        norm = torch.sum(torch.abs(integrated_grad.view(batch_size, -1)), dim=1).view(batch_size, 1, 1, 1)
        # Avoid division by zero
        norm = torch.clamp(norm, min=1e-8)
        accumulated_integrated_grad = momentum * accumulated_integrated_grad + integrated_grad / norm
        # Line 9 algorithm 1 sum
        img = img + alpha * torch.sign(accumulated_integrated_grad)
        img = torch.clamp(img, min=-1, max=1)
    return img, accumulated_integrated_grad, original_image - img

def get_universal_perturbation_mean_gradients(models, imgs, labels, epsilon, iterations, momentum, loss, baseline, preprocessor, order_of_approximation, batch_size,
                                              out_dir = None, name = ""):
    gradients = torch.zeros_like(baseline[0]).cuda()
    total_images = len(imgs)
    num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division

    for batch_idx in range(num_batches):
        start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(total_images, start_idx + batch_size)
        batch_imgs = imgs[start_idx:end_idx]
        print(f"Processing batch {batch_idx + 1}/{num_batches}")
        img_tensor = preprocessor(batch_imgs).cuda()
        # Adjust baseline size to match the actual batch size
        current_baseline = baseline[:img_tensor.size(0)]
        for model in tqdm(models, desc="Attacking models"):
            _, gradient, _ = MIGAttack(model, img_tensor, labels, epsilon, iterations, momentum, loss, current_baseline, preprocessor, order_of_approximation)
            # Sum gradients over the batch dimension
            gradients += gradient.sum(dim=0) / (total_images * len(models))
        end_time = time.time()
        print(f"{batch_idx + 1}/{num_batches} done: " + str(end_time - start_time) + "s")

        if not out_dir is None and batch_idx % 5 == 0:
            # current_grad_img = (gradients.detach().cpu().numpy().transpose(2, 1, 0) * 255).astype(np.uint8)
            # cv2.imwrite(os.path.join(out_dir, name + f"_idx{batch_idx}.png"), current_grad_img)
            torch.save(gradients[0].detach().cpu(), os.path.join(out_dir, name + f"_idx{batch_idx}.pt"))

    return gradients.detach().cpu()

def eval_MIG(input_dir, running_percent, perturbation, model, batch_size):
    start_time = time.time()
    perturbation_sg = (8/255 * torch.sign(perturbation))
    perturbation_np = (perturbation_sg.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    imgs, labels, _ = get_yolo_boxes(input_dir, running_percent, img_size=IMAGE_SIZE)
    total_images = len(imgs)
    num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division
    mapsv8Adv = []
    mapsv8Clean = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(total_images, start_idx + batch_size)
        batch_imgs = imgs[start_idx:end_idx]
        gt_boxes_batch = labels[start_idx:end_idx]

        # Before attack
        results_batch_v8 = model(batch_imgs, verbose=False)
        pred_boxes_batch_v8 = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_batch_v8)
        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v8):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8Clean.append(mAP)

        # Apply perturbation to each image in the batch
        perturbed_imgs = [img + perturbation_np for img in batch_imgs]
        # After attack
        results_batch_v8 = model(perturbed_imgs, verbose=False)
        pred_boxes_batch_v8 = get_predicted_boxes(perturbed_imgs, gt_boxes_batch, results_batch_v8)
        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v8):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8Adv.append(mAP)


        print("Mean adv", np.array(mapsv8Adv).mean())
        print("Mean clean", np.array(mapsv8Clean).mean())

    print("--- %s eval seconds ---" % (time.time() - start_time))
    return np.array(mapsv8Adv).mean(), np.array(mapsv8Clean).mean()

if __name__ == '__main__':
    batch_size = 16

    # Load models
    model_yolo8n = YOLO("models/yolov8n_csgo_apex_aim.pt", verbose=False).cuda()
    model_yolo11n = YOLO('models/yolo11n_csgo_apex_aim.pt', verbose=False).cuda()
    model_yolo5n = YOLO("models/yolov5n_csgo_apex_aim.pt", verbose=False).cuda()
    model_yolo11s = YOLO('models/yolo11s_csgo_apex_aim.pt', verbose=False).cuda()
    model_yolo8s = YOLO("models/yolov8s_csgo_apex_aim.pt", verbose=False).cuda()
    model_yolo5s = YOLO('models/yolov5s_csgo_apex_aim.pt', verbose=False).cuda()

    start_time = time.time()
    running_percent = 1
    loss = BCE_loss_boxes
    baseline = torch.zeros((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)).cuda()
    preprocessor = get_preprocessor(model_yolo8n, batch_size)
    mapsv8Clean = []
    mapsv8Adv = []

    # Training data
    input_dir = "H://blackboxadversarial//dataset//aim_apex//test"
    imgs, labels, _ = get_yolo_boxes(input_dir, running_percent, img_size=IMAGE_SIZE)
    models = [model_yolo8n, model_yolo11n, model_yolo5n, model_yolo11s, model_yolo8s, model_yolo5s]
    perturbation = get_universal_perturbation_mean_gradients(models, imgs[:320], labels, 16/255, 10, 1, loss, baseline, preprocessor, 20, batch_size)
    torch.save(perturbation, "MIG_GRADIENT.pt")
    perturbation = (8/255 * torch.sign(perturbation))
    perturbation_np = (perturbation.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Testing data
    input_dir = "H://blackboxadversarial//dataset//aim_apex//test"
    imgs, labels, _ = get_yolo_boxes(input_dir, running_percent, img_size=IMAGE_SIZE)
    total_images = len(imgs)
    num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(total_images, start_idx + batch_size)
        batch_imgs = imgs[start_idx:end_idx]
        gt_boxes_batch = labels[start_idx:end_idx]
        
        # Apply perturbation to each image in the batch
        perturbed_imgs = [img + perturbation_np for img in batch_imgs]
        
        # After attack
        results_batch_v8 = model_yolo8n(perturbed_imgs, verbose=False)
        pred_boxes_batch_v8 = get_predicted_boxes(perturbed_imgs, gt_boxes_batch, results_batch_v8)
        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v8):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8Adv.append(mAP)
        
        # Before attack
        results_batch_v8 = model_yolo8n(batch_imgs, verbose=False)
        pred_boxes_batch_v8 = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_batch_v8)
        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v8):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8Clean.append(mAP)
        
        print("Mean adv", np.array(mapsv8Adv).mean())
        print("Mean clean", np.array(mapsv8Clean).mean())

