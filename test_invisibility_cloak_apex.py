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

class FindNoise(torch.nn.Module):
    def __init__(self, models,epsilon,noise):
        super(FindNoise,self).__init__()
        self.models = models
        self.epsilon = epsilon
        self.noise = torch.nn.parameter.Parameter(noise)
        self.noise.requires_grad = True

    def forward(self,input,model_idx):
        noise = torch.clamp(self.noise, -self.epsilon, self.epsilon).requires_grad_(True)
        input_data = input.clone().requires_grad_(True)
        adv_img = torch.clamp(input_data + noise, 0.0, 1.0) 
        outputs = self.models[model_idx].model(adv_img)
        preds = ops.non_max_suppression(outputs, conf_thres=0.0, iou_thres=0.0, max_det=25)
        # Determine the maximum number of detections in preds
        max_detections = max([p.shape[0] for p in preds])
        
        # Pad each tensor to have the same number of detections
        padded_preds = []
        for p in preds:
            pad_size = max_detections - p.shape[0]
            if pad_size > 0:
                # Create padding of zeros
                padding = torch.zeros((pad_size, p.shape[1]), device=p.device, dtype=p.dtype)
                p = torch.cat([p, padding], dim=0)
            padded_preds.append(p)
        
        return torch.stack(padded_preds)
    
    def forward_sign(self,input,model_idx):
        noise = torch.clamp(self.noise, -self.epsilon, self.epsilon).requires_grad_(True)
        noise = self.epsilon*torch.sign(self.noise)
        input_data = input.clone().requires_grad_(True)
        adv_img = torch.clamp(input_data + noise, 0.0, 1.0) 
        outputs = self.models[model_idx].model(adv_img)
        preds = ops.non_max_suppression(outputs, conf_thres=0.0, iou_thres=0.0, max_det=25)
        # Determine the maximum number of detections in preds
        max_detections = max([p.shape[0] for p in preds])
        
        # Pad each tensor to have the same number of detections
        padded_preds = []
        for p in preds:
            pad_size = max_detections - p.shape[0]
            if pad_size > 0:
                # Create padding of zeros
                padding = torch.zeros((pad_size, p.shape[1]), device=p.device, dtype=p.dtype)
                p = torch.cat([p, padding], dim=0)
            padded_preds.append(p)
        
        return torch.stack(padded_preds)
    
    def forward_mask(self,input,model_idx):
        input_data = input.clone().requires_grad_(True)
        adv_img = (1-self.noise)*input_data
        outputs = self.models[model_idx].model(adv_img)
        preds = ops.non_max_suppression(outputs, conf_thres=0.0, iou_thres=0.0, max_det=25)
        # Determine the maximum number of detections in preds
        max_detections = max([p.shape[0] for p in preds])
        
        # Pad each tensor to have the same number of detections
        padded_preds = []
        for p in preds:
            pad_size = max_detections - p.shape[0]
            if pad_size > 0:
                # Create padding of zeros
                padding = torch.zeros((pad_size, p.shape[1]), device=p.device, dtype=p.dtype)
                p = torch.cat([p, padding], dim=0)
            padded_preds.append(p)
        
        return torch.stack(padded_preds)
    
    def get_adv(self,input):
        with torch.no_grad():
            noise = torch.clamp(self.noise, -self.epsilon, self.epsilon) 
            adv_img = torch.clamp(input + noise, 0.0, 1.0) 
        return adv_img

    def get_adv_sign(self, input):
        with torch.no_grad():
            thres = 0.01317
            pert = torch.where(self.noise.abs() < thres, torch.tensor(0.0), self.noise)
            noise = self.epsilon*torch.sign(pert)
            # noise = torch.clamp(self.noise, -self.epsilon, self.epsilon) 
            adv_img = torch.clamp(input + noise, 0.0, 1.0) 
        return adv_img
    
    def get_adv_mask(self, input):
        with torch.no_grad():
            adv_img = (1-self.noise)*input
        return adv_img



global_noise=None
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


def attack_success(preds,conf_thres=0.1):
    preds = preds.detach()
    xc = preds[..., 4] > 0.25  # candidates

    x = preds[0][xc[0]]  # confidence
    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    # Box/Mask
    box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
    mask = x[:, 85:]  # zero columns if no masks
    # Detections matrix nx6 (xyxy, conf, cls)
    conf, j = x[:, 5:85].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
    count = x.shape[0]
    return count

def tog_vanishing(x,models,noise_model,optimizer,n_iter=3, eps=8/255., lr=0.001):
    gt_conf = torch.zeros((x.shape[0], 63000, 1),device="cuda")
    t_start = time.time()
    succ = 0
    query = 0
    x = x.clone()


    for k in range(n_iter):
        compute_loss = 0
        x.requires_grad = True
        for idx in range(len(models)):
            preds = noise_model(x,idx)
            compute_loss = BCEobj(preds[...,4:5], gt_conf[:,:preds.shape[1],:]) 
            compute_loss.backward(retain_graph=True)
            # compute loss and backward and update eta
            models[idx].zero_grad()
        optimizer.step()
        optimizer.zero_grad()
    x_adv = noise_model.get_adv(x)
    t = time.time() - t_start
    return x_adv,query,succ,t



IMAGE_SIZE=640

if __name__ == '__main__':
    model_yolo8 = YOLO("models/yolo8n_best.pt", verbose=False)
    model_yolo11 = YOLO('models/yolo11n_best.pt', verbose=False)

    start_time = time.time()
    running_percent = 1
    loss = BCE_loss_boxes
    preprocessor = get_preprocessor(model_yolo8)
    baseline = torch.zeros((1,3,IMAGE_SIZE,IMAGE_SIZE)).cuda()
    mapsv8Clean = []
    mapsv8Adv = []
    input_dir = "H://blackboxadversarial//dataset//aim_apex//test"
    imgs, labels, _ = get_yolo_boxes(input_dir, running_percent, img_size=IMAGE_SIZE)
    # create find noise model 
    eps=16/255.
    lr=0.001
    bs = 32
    pretrained = "MIG_GRADIENT.pt"
    if pretrained:
        eta = torch.load("MIG_GRADIENT.pt").to("cuda")
        print("using pretrained initial gradient")
    else:
        eta = torch.rand((bs, 3, IMAGE_SIZE,IMAGE_SIZE),device="cuda").mul(2*eps).sub(eps)
    eta_start = eta.clone()
    noise_model = FindNoise([model_yolo8, model_yolo11], eps,eta).to("cuda")
    optimizer = torch.optim.Adam([{'params': noise_model.noise}],lr=lr)
    for start_batch_idx in range(0, len(imgs) - 1, bs):
        batch_imgs = imgs[start_batch_idx:start_batch_idx + bs]
        gt_boxes_batch = labels[start_batch_idx:start_batch_idx + bs]
        if (len(batch_imgs)<bs):
            continue
        # after attack
        img = preprocessor(batch_imgs)
        img_adv,_,_,_ = tog_vanishing(img,[model_yolo8, model_yolo11],noise_model, optimizer, n_iter=3, eps=eps, lr=lr)
    print(torch.sum(noise_model.noise-eta_start))

    input_dir = "H://blackboxadversarial//dataset//aim_apex//test"
    imgs, labels, _ = get_yolo_boxes(input_dir, running_percent, img_size=IMAGE_SIZE)
    model = YOLO("models/yolo8n_best.pt", verbose=False)
    for start_batch_idx in range(0, len(imgs) - 1, bs):
        batch_imgs = imgs[start_batch_idx:start_batch_idx + bs]
        gt_boxes_batch = labels[start_batch_idx:start_batch_idx + bs]
        if (len(batch_imgs)<bs):
            continue
        img = preprocessor(batch_imgs)
        results_batch_v8 = model( noise_model.get_adv(img), verbose=False)
        pred_boxes_batch_v8 = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_batch_v8)
        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v8):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8Adv.append(mAP)
        # before attack
        img = preprocessor(batch_imgs)
        results_batch_v8 = model(batch_imgs, verbose=False)
        pred_boxes_batch_v8 = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_batch_v8)
        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v8):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8Clean.append(mAP)
        print("Mean adv", np.array(mapsv8Adv).mean())
        print("Mean clean", np.array(mapsv8Clean).mean())
        model_yolo8.model.train()
    utils.save_image(torch.sqrt(noise_model.noise), "noise.png")
    torch.save(noise_model.noise, "train_yolo8n_yolo11n_train_test.pt")
    input()
    
    # print("--- %s seconds ---" % (time.time() - start_time))
