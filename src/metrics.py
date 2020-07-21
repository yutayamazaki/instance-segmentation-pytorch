from typing import List

import numpy as np
import torch


def box_matching(
    gt_boxes, gt_labels, pred_boxes, pred_labels,
    pred_scores, iou_threshold: float = 0.5, score_threshold: float = 0.0
):
    """Finds matches between prediction and ground truth.
    Args:
        gt_boxes, pred_boxes (List[torch.Tensor]):
            (N, 4) ex) [(x1, y1, x2, y2), ...].
        gt_labels, pred_labels (torch.Tensor): (N, ) ex) [0, 2, 3, 1, 2, ...].
        pred_scores (torch.Tensor): (N, ): .
        iou_threshold, score_threshold (float)
    Returns:
        List[int]: A list of index matched to gt.
                   If there is no matching prediction, index = None.
    """
    # Apply threshold by score threshold
    indices = pred_scores > score_threshold
    pred_scores = pred_scores[indices]
    # Sort predictions by score from high to low
    indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = [pred_boxes[idx] for idx in indices]
    pred_labels = [pred_labels[idx] for idx in indices]
    pred_scores = [pred_scores[idx] for idx in indices]

    gt_match: List[int] = []
    pred_match: List[int] = []
    for i in range(len(gt_boxes)):
        gt_box = gt_boxes[i]  # (4, )
        gt_label = gt_labels[i]  # (1, )

        best_iou: float = -1.
        best_gt_idx: int = -1
        best_pred_idx: int = -1
        for j in range(len(pred_boxes)):
            pred_box = pred_boxes[j]  # (4, )
            pred_label = pred_labels[j]  # (1, )
            if pred_label != gt_label:
                continue

            iou: float = box_iou(gt_box, pred_box)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_gt_idx = j
                best_pred_idx = i

        gt_match.append(best_gt_idx)
        pred_match.append(best_pred_idx)
    return gt_match, pred_match


#################################################
# Metrics for Pixel
#################################################


def mask_iou(gt_mask, pred_mask, eps: float = 1e-8) -> float:
    """Calculate Intersection over Union for each pixel.
    Args:
        gt_mask (toch.Tensor): (H, W)
        pred_mask (torch.Tensor): (H, W)
    Returns:
        float: IoU for mask.
    """
    intersection = float((gt_mask.reshape(-1) * pred_mask.reshape(-1)).sum())
    union = float(gt_mask.sum() + pred_mask.sum() - intersection)
    iou = intersection / (union + eps)
    return iou


def mask_mean_iou(
    gt_masks, pred_masks, threshold: float = 0.5, eps: float = 1e-8
) -> float:
    """Calculate Intersection over Union for each pixel.
    Args:
        gt_masks (toch.Tensor): (N, H, W)
        pred_masks (torch.Tensor): (N, 1, H, W)
    Returns:
        float: Mean IoU for mask.
    """
    pred_masks = pred_masks.squeeze(1)
    pred_masks = (pred_masks > threshold).type(torch.uint8)

    num_masks: float = float(len(gt_masks))
    mean_iou: float = 0.
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        iou: float = mask_iou(gt_mask, pred_mask, eps)
        mean_iou += iou / num_masks
    return mean_iou


def masks_mean_iou(
    gt_bboxes, gt_masks, gt_labels, pred_bboxes, pred_masks, pred_labels,
    pred_scores, iou_threshold: float = 0.5, score_threshold: float = 0.0
) -> float:
    """Compute each ground truth of masks.
    Args:
        gt_bboxes (List[torch.Tensor]): (N_gt, 4).
        pred_bboxes (List[torch.Tensor]): (N_pred, 4).
        gt_masks (List[torch.Tensor]): (N_gt, H, W).
        pred_masks (List[torch.Tensor]): (N_pred, 1, H, W).
        gt_labels (torch.Tensor): (N_gt, ).
        pred_labels (torch.Tensor): (N_pred, ).
        pred_scores (torch.Tensor): (N_pred, ).
        iou_threshold (float): default=0.5. A threshold to match bboxes.
        score_threshold (float): default=0.0
    Returns:
        float: Mean IoU of each ground truth bounding boxes.
    """
    gt_indices, pred_indices = box_matching(
        gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores,
        iou_threshold, score_threshold
    )
    num_masks: int = len(gt_masks)
    mean_iou: float = 0.
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if gt_idx == -1:
            continue

        iou = mask_iou(gt_masks[pred_idx], pred_masks[gt_idx])
        mean_iou += iou / num_masks
    return mean_iou


#################################################
# Metrics for Bounding Boxes
#################################################


def box_iou(a, b) -> float:
    """
    Args:
        a, b (torch.Tensor): (xmin, ymin, xmax, ymax).
    Returns:
        float: IoU of specified boxes.
    """
    # If there is no area duplication.
    if max(a[0], b[0]) - min(a[2], b[2]) >= 0:
        return 0.
    if max(a[1], b[1]) - min(a[3], b[3]) >= 0:
        return 0.

    width_a: float = float(a[2] - a[0])
    width_b: float = float(b[2] - b[0])
    height_a: float = float(a[3] - a[1])
    height_b: float = float(b[3] - b[1])

    area_a: float = height_a * width_a
    area_b: float = height_b * width_b

    overlap_xmin: float = float(max(a[0], b[0]))
    overlap_xmax: float = float(min(a[2], b[2]))
    overlap_width: float = overlap_xmax - overlap_xmin

    overlap_ymin: float = float(max(a[1], b[1]))
    overlap_ymax: float = float(min(a[3], b[3]))
    overlap_height: float = overlap_ymax - overlap_ymin
    intersection: float = overlap_width * overlap_height

    union: float = area_a + area_b - intersection
    return intersection / union


def bboxes_mean_iou(
    gt_bboxes, gt_labels, pred_bboxes, pred_labels,
    pred_scores, iou_threshold: float = 0.5, score_threshold: float = 0.0
) -> float:
    """
    Args:
        gt_bboxes, pred_bboxes (List[torch.Tensor]):
            [(xmin, ymin, xmax, ymax), ...].
        gt_labels, pred_labels (torch.Tensor): (N, ).
        pred_scores (torch.Tensor): (N, ).
        iou_threshold (float): default=0.5. A threshold to match bboxes.
        score_threshold (float): default=0.0
    Returns:
        float: Mean IoU of each ground truth.
    """
    gt_indices, pred_indices = box_matching(
        gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores,
        iou_threshold, score_threshold
    )
    num_bboxes: int = len(gt_bboxes)
    mean_iou: float = 0.
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if gt_idx == -1:
            continue

        iou = box_iou(gt_bboxes[pred_idx], pred_bboxes[gt_idx])
        mean_iou += iou / num_bboxes
    return mean_iou


#################################################
# Mean average precision
#################################################


def compute_mean_average_precision(
    gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores,
    iou_threshold: float = 0.5
):
    """
    Args:
        gt_bboxes, pred_bboxes (List[torch.Tensor]):
            [(xmin, ymin, xmax, ymax), ...].
        gt_labels, pred_labels (torch.Tensor): [0, 2, 1, 3, 0, ...].
        pred_scores (torch.Tensor): [0.9, 0.7, ...].
        iou_threshold (float): default=0.5. A threshold used in box matching.
    Returns:
        float: Calculated mAP.
    Note:
        This function calculate the ap of all boxes, not the average of the
        class-wise ap.
    """
    gt_match, pred_match = box_matching(
        gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores,
        iou_threshold
    )
    gt_match, pred_match = np.array(gt_match), np.array(pred_match)
    precisions = \
        np.cumsum(pred_match > -1, axis=0) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1, axis=0) / len(gt_match)

    # Make precisions to monotonically decreasing.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    num_points: int = 11
    precision_points = np.ones(num_points)
    pr_idx: int = 0
    for idx in range(num_points):
        x: float = idx * 0.1
        if x <= recalls[pr_idx]:
            precision_points[idx] = precisions[pr_idx]
        else:
            if pr_idx != len(precisions) - 1:
                pr_idx += 1
            precision_points[idx] = precisions[pr_idx]

    return float(np.mean(precision_points))
