import glob
import os
from builtins import enumerate
import time
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from utils.metric_utils import calculate_map, calculate_iou


def get_predicted_boxes(batch_imgs, batch_labels, results_batch):
    pred_boxes_batch = []
    display = False
    for i, result in enumerate(results_batch):
        img = batch_imgs[i]
        gt = batch_labels[i]
        pred_boxes_img = []
        if display:
            for gt_idx in range(len(gt)):
                cv2.rectangle(img, (int(gt[gt_idx][0]), int(gt[gt_idx][1])), (int(gt[gt_idx][2]), int(gt[gt_idx][3])),
                              (0, 255, 0), 2)
        for idx_r, result_s in enumerate(result):
            boxes = result_s.boxes
            confidences = result_s.boxes.conf
            for b_idx, box in enumerate(boxes):
                conf = confidences[b_idx]
                if conf > 0.1:
                    skip_box = False
                    xyxy = box.xyxy.detach().cpu().numpy()[0]
                    for already_pred_box in pred_boxes_img:
                        iou = calculate_iou(xyxy, already_pred_box)
                        if iou > 0.3:
                            skip_box = True
                    if not skip_box:
                        pred_boxes_img.append(xyxy)
                        if display:
                            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        if display:
            cv2.imshow("img"+str(idx_r), img)
            cv2.waitKey(0)

        pred_boxes_batch.append(pred_boxes_img)
    return pred_boxes_batch


def get_yolo_boxes(input_dir, running_percent, img_size):
    txt_files = sorted(glob.glob(os.path.join(input_dir, '**/**.txt'), recursive=True))
    txt_files = txt_files[:int(running_percent * len(txt_files))]
    labels = []
    imgs = []
    for label_path in tqdm(txt_files, desc="txt"):
        bbox_lines = []
        if "labels" not in label_path:
            continue
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = list(map(float, line.strip().split()))
                # if len(data) != 5:
                #     continue
                # class_id = int(data[0])
                x_center = int(data[1] * img_size)
                y_center = int(data[2] * img_size)
                b_width = int(data[3] * img_size)
                b_height = int(data[4] * img_size)
                bbox_line = [int(x_center - b_width // 2), int(y_center - b_height // 2), int(x_center + b_width // 2),
                             int(y_center + b_height // 2)]
                bbox_lines.append(bbox_line)

        if len(bbox_lines) > 0:
            jpg_path = label_path.replace(".txt", ".jpg").replace("labels", "images")
            img = cv2.imread(jpg_path)
            imgs.append(img)
            # for bbox in bbox_lines:
            #     print(bbox)
            #     cv2.rectangle(img, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4])), (0, 255, 0), 2)
            #     cv2.imshow("img", img)
            #     cv2.waitKey(0)
            labels.append(bbox_lines)

    return imgs, labels, txt_files


def eval_yolo(model_yolo8, model_yolo10, imgs, labels, bs=16):
    start_time = time.time()
    mapsv8 = []
    mapsv11 = []
    # for start_batch_idx in tqdm(range(0, len(imgs) - 1, bs), desc="fitness"):
    for start_batch_idx in range(0, len(imgs) - 1, bs):
        batch_imgs = imgs[start_batch_idx:start_batch_idx + bs]
        gt_boxes_batch = labels[start_batch_idx:start_batch_idx + bs]
        results_batch_v8 = model_yolo8(batch_imgs, verbose=False)
        results_batch_v10 = model_yolo10(batch_imgs, verbose=False)
        pred_boxes_batch_v8 = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_batch_v8)
        pred_boxes_batch_v10 = get_predicted_boxes(batch_imgs, gt_boxes_batch, results_batch_v10)

        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v8):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv8.append(mAP)

        for img_idx_in_batch, pred_boxes in enumerate(pred_boxes_batch_v10):
            gt_boxes = gt_boxes_batch[img_idx_in_batch]
            mAP, precisions, recalls = calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5])
            mapsv11.append(mAP)

    mean_map_v8 = np.array(mapsv8).mean()
    mean_map_v11 = np.array(mapsv11).mean()
    # print('Mean MAP v8: {:.3f}'.format(mean_map_v8))
    # print('Mean MAP v11: {:.3f}'.format(mean_map_v11))
    # print("--- %s eval time ---" % (time.time() - start_time))
    return (mean_map_v8 + mean_map_v11) / 2, time.time() - start_time


if __name__ == '__main__':
    model_yolo8 = YOLO("models/rtdetr-l_csgo_apex_aim2.pt", verbose=False)
    model_yolo11 = YOLO('models/yolo11n_best.pt', verbose=False)


    # start_time = time.time()
    # metrics = model_yolo8.val(data="experiments/val_small.yaml", imgsz=640, batch=16, workers=4, verbose=False,
    #                           half=False, device="cuda", split="val",plots=False)
    # print(" map50-95",metrics.box.map)
    # print(" map50",metrics.box.map50)  # map50
    # print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    input_dir = "D:/work/image_databases/adversarialAttack/aim/test"
    running_percent = 1
    imgs, labels, txt_files = get_yolo_boxes(input_dir, running_percent, img_size=640)

    mean_map = eval_yolo(model_yolo8, model_yolo11, imgs, labels, 16)
    print("mean_map", mean_map)
    # print("--- %s seconds ---" % (time.time() - start_time))
