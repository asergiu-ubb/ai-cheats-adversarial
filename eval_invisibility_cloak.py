import glob
import os
from builtins import enumerate
import time
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch
from utils.metric_utils import calculate_map, calculate_iou
from eval_yolo import get_yolo_boxes, get_predicted_boxes, calculate_map
from ultralytics.utils import ops
from torchvision import utils
from PIL import Image
from MIG import BCE_loss_boxes, get_preprocessor


BCEobj = torch.nn.BCELoss()
BCElogits = torch.nn.BCEWithLogitsLoss()
gt_conf = torch.zeros((32, 63000, 1),device="cuda")
gt_conf_nano = torch.zeros((1, 21250),device="cuda")

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom rightog_vanishingt x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y



def our_attack_success(preds, conf_thres, class_idx=0):
    batch_count = 0
    for boxes in preds:
        if boxes.numel() == 0:
            continue  # Skip empty tensors
        # Extract the confidence scores (5th column) and class labels (6th column)
        confidences = boxes[:, 4]
        classes = boxes[:, 5]
        high_conf_and_class = (confidences > conf_thres)
        # Check if any boxes meet the criteria in this batch
        if high_conf_and_class.any():
            batch_count += 1
    return batch_count

def tog_vanishing_with_update(xs,train_model_idx,noise_model,optimizer,n_iter=1, eps=8/255., lr=0.001, noise_function=None, only_universal=False):
    gt_conf = torch.zeros((xs.shape[0], 63000, 1),device="cuda")
    t_start = time.time()
    succ = 0
    query = 0
    x = xs.clone()
    if noise_function == "get_adv_mask":
        get_noise = noise_model.get_adv_mask
    elif noise_function == "get_adv":
        get_noise = noise_model.get_adv
    else:
        get_noise = noise_model.get_adv_sign
    # test global noise
    with torch.no_grad():
        outputs = noise_model(x,train_model_idx)
        query += 1
        result = our_attack_success(outputs, 0.4)
        if result == 0: # attack success
            succ = 1
            t = time.time() - t_start
            return get_noise(x),query,succ,t
    if only_universal:
        return get_noise(x),query,succ,0


    for k in range(n_iter):
        compute_loss = 0
        x.requires_grad = True
        preds = noise_model(x,train_model_idx)
        compute_loss = BCEobj(preds[...,4:5], gt_conf[:,:preds.shape[1],:]) 
        compute_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        noise_model.models[train_model_idx].zero_grad()
        query += 1
        result = our_attack_success(preds, 0.4)
        if result == 0:
            succ = 1
            break
    x_adv = get_noise(x)

    
    t = time.time() - t_start
    return x_adv,query,succ,t

IMAGE_SIZE=640

if __name__ == '__main__':
    model_yolo8 = YOLO("models/yolo8n_best.pt", verbose=False).cuda()
    model_yolo11 = YOLO('models/yolo11n_best.pt', verbose=False).cuda()

    start_time = time.time()
    running_percent = 1
    loss = BCE_loss_boxes
    preprocessor = get_preprocessor(model_yolo8)
    baseline = torch.zeros((1,3,IMAGE_SIZE,IMAGE_SIZE)).cuda()
    mapsv8Clean = []
    mapsv8Adv = []
    successAdv = []
    queriesAdv = []
    # create find noise model 
    eps=16/255.
    lr=0.01
    bs = 32
    train = True
    eta = torch.load("train_yolo8n_yolo11n_train_test.pt").to("cuda")
    #eta = torch.rand((bs, 3, IMAGE_SIZE,IMAGE_SIZE),device="cuda").mul(2*eps).sub(eps)
    noise_model = FindNoise([model_yolo8, model_yolo11], eps,eta).to("cuda")
    optimizer = torch.optim.Adam([{'params': noise_model.noise}],lr=lr)

    input_dir = "H://blackboxadversarial//dataset//aim_apex//test"
    imgs, labels, _ = get_yolo_boxes(input_dir, running_percent, img_size=IMAGE_SIZE)
    for start_batch_idx in range(0, len(imgs) - 1, bs):
        batch_imgs = imgs[start_batch_idx:start_batch_idx + bs]
        gt_boxes_batch = labels[start_batch_idx:start_batch_idx + bs]
        if (len(batch_imgs)<bs):
            continue
        img = preprocessor(batch_imgs)
        img_adv, queries, succ, t = tog_vanishing_with_update(img, 0, noise_model, optimizer,eps=eps,lr=lr)
        
        
        #add check success for target model
        queriesAdv.append(queries)
        
        # check success rate for target model
        outputs = model_yolo11.model(img_adv)
        outputs = ops.non_max_suppression(outputs, conf_thres=0.0, iou_thres=0.0, max_det=25)
        outputs = torch.stack(outputs) 
        result = our_attack_success(outputs, 0.4)
        successAdv.append(result/bs)
        
        results_batch_target = model_yolo11( img_adv, verbose=False)
        pred_boxes_batch_target = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_batch_target)
        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_target):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8Adv.append(mAP)
        # before attack
        results_batch_v8 = model_yolo11(batch_imgs, verbose=False)
        pred_boxes_batch_v8 = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_batch_v8)
        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v8):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8Clean.append(mAP)
        print("Mean adv", np.array(mapsv8Adv).mean())
        print("Mean clean", np.array(mapsv8Clean).mean())
        print("Mean success rate", np.array(successAdv).mean())
        print("Mean queries", np.array(queriesAdv).mean())
    #utils.save_image(torch.sqrt(noise_model.noise), "noise.png")
    #torch.save(noise_model, "find_noise_yolo8n_apex_train.pt")
    input()
    
    # print("--- %s seconds ---" % (time.time() - start_time))
