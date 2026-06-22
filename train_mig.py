import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import argparse


from MIG import get_preprocessor, BCE_loss_boxes, get_universal_perturbation_mean_gradients, eval_MIG
from eval_yolo import get_yolo_boxes


IMAGE_SIZE = 640


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("eval_root_path")
    args = parser.parse_args()
    input_dir = args.path
    eval_root_dir = args.eval_root_path

    model_yolo8n = YOLO("models/yolov8n_csgo_apex_aim.pt", verbose=False).cuda()
    model_yolo11n = YOLO('models/yolo11n_csgo_apex_aim.pt', verbose=False).cuda()
    model_yolo5n = YOLO("models/yolov5n_csgo_apex_aim.pt", verbose=False).cuda()
    model_yolo11s = YOLO('models/yolo11s_csgo_apex_aim.pt', verbose=False).cuda()
    model_yolo8s = YOLO("models/yolov8s_csgo_apex_aim.pt", verbose=False).cuda()
    model_yolo5s = YOLO('models/yolov5s_csgo_apex_aim.pt', verbose=False).cuda()

    current_datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = f"MIG_res/{current_datetime_str}/"
    os.makedirs(out_dir, exist_ok=True)

    BATCH_SIZE = 64
    start_time = time.time()
    running_percent = 1
    iterations = 10
    order_of_approximation = 20
    loss = BCE_loss_boxes
    name = f"MIG_GRADIENT_ds{running_percent}_its{iterations}_order{order_of_approximation}"

    preprocessor = get_preprocessor(model_yolo8n, BATCH_SIZE)
    baseline = torch.zeros((1,3,IMAGE_SIZE,IMAGE_SIZE)).cuda()

    imgs, labels, _ = get_yolo_boxes(input_dir, running_percent, img_size=IMAGE_SIZE)
    models = [model_yolo8n, model_yolo11n, model_yolo5n, model_yolo11s, model_yolo8s, model_yolo5s]
    perturbation = get_universal_perturbation_mean_gradients(models, imgs, labels, 16/255, iterations, 1, loss, baseline, preprocessor,
                                                             order_of_approximation, BATCH_SIZE, out_dir, name=name)

    final_perturbation_path = os.path.join(out_dir, f"{name}_final.pt")

    torch.save(perturbation, final_perturbation_path)
    pert_display = (perturbation.detach().numpy() * 255).astype(np.uint8).transpose(2, 1, 0)
    cv2.imwrite(os.path.join(out_dir, f"{name}_final.png"), pert_display)
    eval_dirs = [os.path.join(eval_root_dir,"aim/test"), os.path.join(eval_root_dir,"CSGO/test"), os.path.join(eval_root_dir,"apex/test"),
                 os.path.join(eval_root_dir,"aim/valid"), os.path.join(eval_root_dir,"CSGO/valid"), os.path.join(eval_root_dir,"apex/valid")]

    for eval_dir in eval_dirs:
        mapsv8Adv, mapsv8Clean = eval_MIG(eval_dir, 1,perturbation, model_yolo8n, BATCH_SIZE)
        with open(os.path.join(out_dir,"mig_res.txt"), "a") as f:
            f.write(f"Mean adv model_yolo8n {np.array(mapsv8Adv).mean()}\n")
            f.write(f"Mean clean model_yolo8n {np.array(mapsv8Clean).mean()}\n")

        mapsv8Adv, mapsv8Clean = eval_MIG(eval_dir, 1, perturbation, model_yolo11s, BATCH_SIZE)
        with open(os.path.join(out_dir,"mig_res.txt"), "a") as f:
            f.write(f"Mean adv model_yolo11s {np.array(mapsv8Adv).mean()}\n")
            f.write(f"Mean clean model_yolo11s {np.array(mapsv8Clean).mean()}\n")
        print(f"write result to {os.path.join(out_dir, 'mig_res.txt')}")

