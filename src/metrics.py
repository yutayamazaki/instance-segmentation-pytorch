from typing import List, Tuple

import numpy as np
import torch


def box_matching(
    gt_bboxes, gt_labels, pred_bboxes, pred_labels,
    pred_scores, iou_threshold: float = 0.5, score_threshold: float = 0.0
) -> Tuple[List[int], List[int]]:
    """Find matches of bounding boxes between prediction and ground truth.
    Args:
        gt_bboxes (List[torch.Tensor]): (N_gt, 4) ex) [(x1, y1, x2, y2), ...].
        pred_bboxes (List[torch.Tensor]): (N_pred, 4).
        gt_labels (torch.Tensor): (N_gt, ). ex) [0, 2, 3, 1, 2, ...].
        pred_labels (torch.Tensor): (N_pred, ). ex) [0, 2, 3, 1, 2, ...].
        pred_scores (torch.Tensor): (N_pred, ). ex) [0.9, 0.8, ...].
        iou_threshold (float): Threshold used to match bboxes.
        score_treshold (float): A minimum threshold for pred_scores.
    Returns:
        Tuple[List[int], List[int]]: A list of index matched to gt and pred.
    """
    # Filter and remove predictions by score_threshold.
    indices = pred_scores > score_threshold
    pred_scores = pred_scores[indices]
    # Sort predictions by score from high to low.
    indices = torch.argsort(pred_scores, descending=True)
    pred_bboxes = [pred_bboxes[idx] for idx in indices]
    pred_labels = [pred_labels[idx] for idx in indices]
    pred_scores = [pred_scores[idx] for idx in indices]

    gt_match: List[int] = []
    pred_match: List[int] = []
    for gt_i, (gt_bbox, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
        best_iou: float = -1.
        best_gt_idx: int = -1
        best_pred_idx: int = -1
        for pred_j, (pred_bbox, pred_label) in enumerate(
            zip(pred_bboxes, pred_labels)
        ):
            if pred_label != gt_label:
                continue

            iou: float = bbox_iou(gt_bbox, pred_bbox)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_gt_idx = pred_j
                best_pred_idx = gt_i

        gt_match.append(best_gt_idx)
        pred_match.append(best_pred_idx)
    return gt_match, pred_match


#################################################
# Metrics for Pixel
#################################################


def mask_iou(gt_mask, pred_mask, eps: float = 1e-8) -> float:
    """Calculate Intersection over Union for mask.
    Args:
        gt_mask (toch.Tensor): (H, W)
        pred_mask (torch.Tensor): (H, W)
    Returns:
        float: Calculated IoU of given two masks.
    """
    intersection = (gt_mask.reshape(-1) * pred_mask.reshape(-1)).sum()
    union = gt_mask.sum() + pred_mask.sum() - intersection
    iou = intersection / (union + eps)
    return float(iou)


def masks_mean_iou(
    gt_bboxes, gt_masks, gt_labels, pred_bboxes, pred_masks, pred_labels,
    pred_scores, iou_threshold: float = 0.5, score_threshold: float = 0.0
) -> float:
    """Compute masks IoU between ground truth and predictions.
    Args:
        gt_bboxes (List[torch.Tensor]): (N_gt, 4).
        pred_bboxes (List[torch.Tensor]): (N_pred, 4).
        gt_masks (torch.Tensor): (N_gt, H, W).
        pred_masks (torch.Tensor): (N_pred, 1, H, W).
        gt_labels (torch.Tensor): (N_gt, ).
        pred_labels (torch.Tensor): (N_pred, ).
        pred_scores (torch.Tensor): (N_pred, ).
        iou_threshold (float): default=0.5. A threshold used to match bboxes.
        score_threshold (float): default=0.0. Minimum threshold to filter
                                 predictions.
    Returns:
        float: Mean IoU of given masks.
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
        mean_iou += iou
    return mean_iou / num_masks


#################################################
# Metrics for Bounding Boxes
#################################################


def bbox_iou(a, b) -> float:
    """
    Args:
        a, b (torch.Tensor): Bounding box (xmin, ymin, xmax, ymax).
    Returns:
        float: IoU of given two bboxes.
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
        gt_bboxes: (LIst[torch.Tensor]): (N_gt, 4).
                                         [(xmin, ymin, xmax, ymax), ...].
        pred_bboxes (List[torch.Tensor]): (N_pred, 4).
        gt_labels: (torch.Tensor): (N_gt, ).
        pred_labels (torch.Tensor): (N_pred, ).
        pred_scores (torch.Tensor): Confidence scores with shape (N_pred, ).
        iou_threshold (float): default=0.5. A threshold used to match bboxes.
        score_threshold (float): default=0.0. Minimun threshold to filter
                                 predictions.
    Returns:
        float: Mean IoU of given bounding boxes.
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

        iou = bbox_iou(gt_bboxes[pred_idx], pred_bboxes[gt_idx])
        mean_iou += iou
    return mean_iou / num_bboxes


#################################################
# Mean average precision
#################################################


def compute_mean_average_precision(
    gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores,
    iou_threshold: float = 0.5
):
    """
    Args:
        gt_bboxes (List[torch.Tensor]): (N_gt, 4) like
                                        [(xmin, ymin, xmax, ymax), ...].
        pred_bboxes (List[torch.Tensor]): (N_pred, 4).
        gt_labels (torch.Tensor): (N_gt, ).
        pred_labels (torch.Tensor): (N_pred, ).
        pred_scores (torch.Tensor): COnfidence score with shape (N_pred, ).
        iou_threshold (float): default=0.5. A threshold used in box matching.
    Returns:
        float: Calculated mAP.
    Note:
        This function calculate the ap of all boxes, not the average of the
        class-wise ap.
    """
    gt_match_list, pred_match_list = box_matching(
        gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores,
        iou_threshold
    )
    gt_match = np.array(gt_match_list)
    pred_match = np.array(pred_match_list)
    precisions = \
        np.cumsum(pred_match > -1, axis=0) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1, axis=0) / len(gt_match)

    # Make precisions monotonically decreasing.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Calculate mean each 11 points of precisions.
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
