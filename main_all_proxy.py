import argparse
import os
from datetime import datetime

import torch
import numpy as np
import time
import logging
from torchvision import utils
from ultralytics import YOLO


from MIG import BCE_loss_boxes, get_preprocessor, get_universal_perturbation_mean_gradients
from test_invisibility_cloak_apex import FindNoise, tog_vanishing
from eval_invisibility_cloak import tog_vanishing_with_update, our_attack_success
from eval_yolo import get_yolo_boxes, get_predicted_boxes, calculate_map
from ultralytics.utils import ops
from copy import deepcopy 
from ultralytics import RTDETR

IMAGE_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def join_basenames(paths):
    # Extract basenames from paths and join them with "_"
    basenames = [os.path.basename(os.path.dirname(path)) for path in paths]
    return "_".join(basenames)


def main(args):
    # Set up logging
    log_file = args.log_file
    log_file= log_file.replace("/log.txt", f'{args.experiment}/[init]_{args.initial_perturbation}'
                                 f'_[pert]_{args.perturbation_mode}'
                                 f'_[only_rtdets]_{args.only_rtdetr}'
                                 f'_[train_dataset]_{join_basenames(args.train_datasets)}'
                                 f'_[test_dataset]_{join_basenames(args.test_datasets)}'
                                 f'_[train]_{os.path.basename(args.train_model).replace("_all.pt", "")}/log.txt')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # Start of the experiment
    start_time = time.time()
    logger.info("Experiment started.")
    logger.info(f"Arguments: {args}")

    # Set up models
    logger.info("Loading models...")
    models_paths = [
        "models/rtdetr-x_640_all.pt",
        "models/rtdetr-l_640_all.pt",
        "models/yolo8n_640_all.pt",
        "models/yolo8s_640_all.pt",
        "models/yolo11n_640_all.pt",
        "models/yolo11s_640_all.pt",
        "models/yolo5s_640_all.pt",
        "models/yolo5n_640_all.pt"
    ]

    all_models = []
    train_idx = 0
    for m_idx, model_path in enumerate(models_paths):
        logger.info(f"Loading {model_path}...")
        if "yolo" in model_path:
            model = YOLO(model_path, verbose=False).to(DEVICE)
        else:
            model = RTDETR(model_path).to(DEVICE)
        all_models.append(model)
        if os.path.basename(model_path) == os.path.basename(args.train_model):
            train_idx = m_idx
    logger.info(f"Found train idx {train_idx}.")

    # Set up preprocessor
    batch_size = args.batch_size
    preprocessor = get_preprocessor(all_models[0], batch_size)
    baseline = torch.zeros((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
    loss_fn = BCE_loss_boxes

    # Load datasets
    logger.info("Loading datasets...")
    running_percent = 1.0 # Adjust as needed
    train_imgs, train_labels = [], []
    for dataset_path in args.train_datasets:
        imgs, labels, _ = get_yolo_boxes(dataset_path, running_percent, img_size=IMAGE_SIZE)
        train_imgs.extend(imgs)
        train_labels.extend(labels)
    test_imgs, test_labels = [], []
    for dataset_path in args.test_datasets:
        imgs, labels, _ = get_yolo_boxes(dataset_path, running_percent, img_size=IMAGE_SIZE)
        test_imgs.extend(imgs)
        test_labels.extend(labels)

    # Generate or load initial perturbation
    epsilon = args.epsilon / 255.0
    if args.initial_perturbation_path:
        logger.info(f"Loading initial perturbation from {args.initial_perturbation_path}...")
        initial_perturbation = torch.load(args.initial_perturbation_path).to(DEVICE)
    elif args.initial_perturbation.lower() == 'mig':
        logger.info("Generating initial perturbation using MIG...")
        iterations = 10
        momentum = 1
        order_of_approximation = 20
        initial_perturbation = get_universal_perturbation_mean_gradients(
            all_models, train_imgs, train_labels, epsilon, iterations, momentum, loss_fn, baseline, preprocessor, order_of_approximation, batch_size)
        initial_perturbation = initial_perturbation.to(DEVICE)
    else:  # Random initial perturbation
        logger.info("Generating random initial perturbation...")
        initial_perturbation = torch.rand((3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE).mul(2 * epsilon).sub(epsilon)
    # Save the trained perturbation if needed
    if args.save_perturbation:
        init_perturbation_path = f"{log_file.replace('.log','')}_init_perturbation.pt"
        torch.save(initial_perturbation, init_perturbation_path)
        logger.info(f"Trained perturbation saved to {init_perturbation_path}")
    # Set up the noise model
    logger.info("Setting up the noise model...")
    noise_model = FindNoise(all_models, epsilon, initial_perturbation).to(DEVICE)
    optimizer = torch.optim.Adam([{'params': noise_model.noise}], lr=0.001)
    
    if not args.only_test:
        # Train the FindNoise model using tog_vanishing
        logger.info("Training the noise model using tog_vanishing...")
        train_batches = len(train_imgs) // batch_size
        for batch_idx in range(train_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_imgs = train_imgs[start_idx:end_idx]
            if len(batch_imgs) < batch_size:
                break  # Skip incomplete batch
            img_tensor = preprocessor(batch_imgs).to(DEVICE)
            # Train using tog_vanishing
            img_adv, query, succ, t = tog_vanishing(img_tensor, all_models, noise_model, optimizer, n_iter=args.num_iterations, eps=epsilon, lr=args.learning_rate)
            logger.info(f"Training batch {batch_idx + 1}/{train_batches} completed.")
        
        # Save the trained perturbation if needed
        if args.save_perturbation:
            perturbation_path = f"{log_file.replace('.log','')}_trained_perturbation.pt"
            torch.save(noise_model.noise, perturbation_path)
            logger.info(f"Trained perturbation saved to {perturbation_path}")
    else:
        logger.info("Skipping training")
        
    for train_model_idx in range(len(models_paths)):
        for test_idx, test_model_path in enumerate(models_paths):
            if args.only_rtdetr:
                if ("yolo" in models_paths[train_model_idx].lower()) and ("rtdetr" not in test_model_path.lower()):
                    continue
            if "yolo" in test_model_path:
                test_model = YOLO(test_model_path, verbose=False).to(DEVICE)
            else:
                test_model = RTDETR(test_model_path).to(DEVICE)
            # Evaluation using tog_vanishing_with_update
            logger.info(f"Evaluating on test data using tog_vanishing_with_update THE MODEL{test_model_path}...")
            maps_clean = []
            maps_adv = []
            total_images = len(test_imgs)
            num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division
            success_rates = []
            queries_list = []
        
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(total_images, start_idx + batch_size)
                batch_imgs = test_imgs[start_idx:end_idx]
                gt_boxes_batch = test_labels[start_idx:end_idx]
        
                if len(batch_imgs) < batch_size:
                    # Padding the batch to have the required batch size
                    padding_size = batch_size - len(batch_imgs)
                    batch_imgs.extend(batch_imgs[:padding_size])
                    gt_boxes_batch.extend(gt_boxes_batch[:padding_size])
        
                img_tensor = preprocessor(batch_imgs).to(DEVICE)
                noise_model_copy = FindNoise(all_models, epsilon, noise_model.noise.clone().detach()).to(DEVICE)
                optimizer = torch.optim.Adam([{'params': noise_model_copy.noise}], lr=0.01)
                # Generate adversarial examples using tog_vanishing_with_update
                img_adv, queries, succ, t = tog_vanishing_with_update(img_tensor, train_model_idx, noise_model_copy, optimizer, n_iter=args.num_iterations, eps=epsilon, lr=args.learning_rate, noise_function = args.perturbation_mode)
                queries_list.append(queries)
                if args.perturbation_mode == "get_adv":
                    preds = noise_model_copy(img_tensor,test_idx)
                else:
                    preds = noise_model_copy.forward_sign(img_tensor,test_idx)
                result = our_attack_success(preds, 0.4)
                if result == 0:
                    success_rates.append(1)
                else:
                    success_rates.append(0)
        
                # Evaluate on test model
                results_adv = test_model(img_adv, verbose=False)
                pred_boxes_adv = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_adv)
                results_clean = test_model(batch_imgs, verbose=False)
                pred_boxes_clean = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_clean)
        
                for img_idx_in_batch in range(len(batch_imgs)):
                    gt_boxes = gt_boxes_batch[img_idx_in_batch]
        
                    # After attack
                    pred_boxes = pred_boxes_adv[img_idx_in_batch]
                    mAP, _, _ = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
                    maps_adv.append(mAP)
        
                    # Before attack
                    pred_boxes = pred_boxes_clean[img_idx_in_batch]
                    mAP, _, _ = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
                    maps_clean.append(mAP)
        
                
            # Final results
            total_time = time.time() - start_time
            logger.info(f"Experiment completed. FINAL RESULTS FOR MODEL TRAIN MODEL:{models_paths[train_model_idx]}  TEST MODEL: {test_model_path}")
            logger.info(f"Final Mean mAP (Clean): {np.mean(maps_clean):.4f}")
            logger.info(f"Final Mean mAP (Adv): {np.mean(maps_adv):.4f}")
            logger.info(f"Average Queries per Image: {np.mean(queries_list):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified parameters.")
    parser.add_argument("--train-datasets", nargs='+', required=True, help="List of paths for datasets to be used for training.")
    parser.add_argument("--test-datasets", nargs='+', required=True, help="List of paths for datasets to be used for testing.")
    parser.add_argument("--initial-perturbation", choices=['MIG', 'random'], required=True, help="Initial perturbation for NoiseModel.")
    parser.add_argument("--perturbation-mode", choices=['get_adv', 'get_adv_sign'], required=True, help="Perturbation mode to use during testing.")
    parser.add_argument("--train-model", required=True, help="Model to be run for tog_vanishing_with_update.")
    parser.add_argument("--initial-perturbation-path", help="Path to the initial perturbation if reusing one already trained.")
    parser.add_argument("--epsilon", type=float, default=8, help="Epsilon value for the perturbation (in pixel intensity units).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing images.")
    parser.add_argument("--num-iterations", type=int, default=3, help="Number of iterations for tog_vanishing and tog_vanishing_with_update.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--save-perturbation", action='store_true', help="Save the generated perturbation.")
    parser.add_argument("--only-test", action='store_true', help="Only test the perturbations")
    parser.add_argument("--log-file", default=f"runs/"+"/log.txt", help="Path to the log file.")
    parser.add_argument("--only_rtdetr", action='store_true', help="Only test the rtdetr")
    parser.add_argument("--experiment")
    args = parser.parse_args()

    main(args)