import numpy as np
import torch.nn as nn
from pytorch_msssim import ssim

# Define the SSIM metric wrapper as a loss function
class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(SSIMLoss, self).__init__()
        # self.ssim = ssim(data_range=data_range, size_average=size_average)

    def forward(self, y_pred, y_true):
        ssim_score = ssim(y_pred* 255, y_true* 255, data_range=255, size_average=True)
        return 1 - ssim_score  # (lower is better)

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Each box is defined as [x_min, y_min, x_max, y_max].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area
    unionArea = boxAArea + boxBArea - interArea

    # Compute IoU
    iou = interArea / float(unionArea)

    return iou


def match_bboxes(gt_boxes, pred_boxes, iou_threshold):
    """
    Match predicted bounding boxes to ground truth boxes using IoU and a given threshold.
    Returns:
        TP, FP, FN counts
    """
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

    # Calculate IoU for each GT-Prediction pair
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt, pred)

    # Initialize counters
    TP = 0
    FP = 0
    FN = 0

    # Keep track of which GT boxes and predicted boxes are already matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    pred_matched = np.zeros(len(pred_boxes), dtype=bool)

    # Match predictions to GT boxes using IoU
    for i in range(len(gt_boxes)):
        if len(iou_matrix) > 0 and len(iou_matrix[0]) > 0:
            max_iou_idx = np.argmax(iou_matrix[i, :])
            max_iou = iou_matrix[i, max_iou_idx]

            if max_iou >= iou_threshold and not pred_matched[max_iou_idx]:
                TP += 1
                gt_matched[i] = True
                pred_matched[max_iou_idx] = True
            else:
                FN += 1
        # else:
        #     FN += 1
    # Count remaining false positives (predicted boxes not matched to any GT)
    FP = np.sum(~pred_matched)
    return TP, FP, FN


def calculate_precision_recall(tp, fp, fn):
    """
    Calculate precision and recall based on true positives, false positives, and false negatives.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


def calculate_map(gt_boxes, pred_boxes, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculate mAP (Mean Average Precision) over IoU thresholds.

    Args:
        gt_boxes (list of list): List of ground truth bounding boxes [[x_min, y_min, x_max, y_max], ...]
        pred_boxes (list of list): List of predicted bounding boxes [[x_min, y_min, x_max, y_max], ...]
        iou_thresholds (list): List of IoU thresholds (default 0.5 to 0.95 with step 0.05).

    Returns:
        mAP: Mean average precision
    """
    precisions = []
    recalls = []

    for iou_thresh in iou_thresholds:
        tp, fp, fn = match_bboxes(gt_boxes, pred_boxes, iou_thresh)
        precision, recall = calculate_precision_recall(tp, fp, fn)
        precisions.append(precision)
        recalls.append(recall)

    # Calculate mAP as the mean of the precisions over all thresholds
    return np.mean(precisions), precisions, recalls