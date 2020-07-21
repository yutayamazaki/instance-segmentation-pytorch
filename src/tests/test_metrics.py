import unittest
from typing import List

import torch

import metrics


class BoxMatchingTests(unittest.TestCase):

    def test_simple(self):
        gt_bboxes: List[torch.Tensor] = [
            torch.Tensor([0, 0, 10, 10]),
            torch.Tensor([20, 20, 30, 30]),
        ]
        pred_bboxes: List[torch.Tensor] = [
            torch.Tensor([1, 2, 11, 9]),
            torch.Tensor([16, 19, 31, 30]),
        ]
        gt_labels = torch.ones(2)
        pred_labels = torch.ones(2)
        pred_scores = torch.Tensor([0.9, 0.8])

        gt_match, pred_match = metrics.box_matching(
            gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores
        )

        self.assertEqual(gt_match, [0, 1])
        self.assertEqual(pred_match, [0, 1])


class MaskIoUTests(unittest.TestCase):

    def test_iou_zero(self):
        gt_mask: torch.Tensor = torch.zeros((3, 3))
        pred_mask: torch.Tensor = torch.zeros((3, 3))

        iou = metrics.mask_iou(gt_mask, pred_mask)
        self.assertIsInstance(iou, float)
        self.assertAlmostEqual(iou, 0.0)

    def test_iou_one(self):
        gt_mask: torch.Tensor = torch.zeros((3, 3))
        pred_mask: torch.Tensor = torch.zeros((3, 3))

        gt_mask[0, 0] = 1
        pred_mask[0, 0] = 1

        iou = metrics.mask_iou(gt_mask, pred_mask)
        self.assertIsInstance(iou, float)
        self.assertAlmostEqual(iou, 1.0)


class MaskMeanIoUTests(unittest.TestCase):

    def test_simple(self):
        gt_masks: List[torch.Tensor] = torch.zeros((2, 3, 3))
        pred_masks: List[torch.Tensor] = torch.zeros((2, 3, 3))

        iou = metrics.mask_mean_iou(gt_masks, pred_masks)
        self.assertIsInstance(iou, float)
        self.assertAlmostEqual(iou, 0.0)


class BoxIoUTests(unittest.TestCase):

    def test_same_bbox_equal_to_one(self):
        # (xmin, ymin, xmax, ymax)
        bbox: torch.Tensor = torch.Tensor([0, 10, 10, 20])
        iou: float = metrics.box_iou(bbox, bbox)

        self.assertIsInstance(iou, float)
        self.assertAlmostEqual(iou, 1.)

    def test_iou_zero(self):
        # (xmin, ymin, xmax, ymax)
        bbox_a: torch.Tensor = torch.Tensor([0, 10, 10, 20])
        bbox_b: torch.Tensor = torch.Tensor([10, 20, 20, 25])
        iou: float = metrics.box_iou(bbox_a, bbox_b)

        self.assertAlmostEqual(iou, 0.)


class BoxesMeanIoUTests(unittest.TestCase):

    def test_simple(self):
        # (xmin, ymin, xmax, ymax)
        gt_boxes: List[torch.Tensor] = [
            torch.Tensor([0, 10, 10, 20]),
            torch.Tensor([0, 10, 10, 20]),
        ]
        pred_boxes: List[torch.Tensor] = [
            torch.Tensor([0, 10, 10, 20]),
            torch.Tensor([0, 10, 10, 20]),
        ]
        iou: float = metrics.boxes_mean_iou(gt_boxes, pred_boxes)

        self.assertIsInstance(iou, float)
        self.assertAlmostEqual(iou, 1.0)


def get_data():
    gt_bboxes: List[torch.Tensor] = [
        torch.Tensor([0, 10, 10, 20]),
        torch.Tensor([0, 10, 10, 20]),
    ]
    pred_bboxes: List[torch.Tensor] = [
        torch.Tensor([0, 10, 10, 20]),
        torch.Tensor([0, 10, 10, 20]),
    ]
    gt_labels: torch.Tensor = torch.ones(2)
    pred_labels: torch.Tensor = torch.ones(2)
    pred_scores: torch.Tensor = torch.Tensor([0.9, 0.8])
    return gt_bboxes, pred_bboxes, gt_labels, pred_labels, pred_scores


class ComputeMeanAveragePrecisionTests(unittest.TestCase):

    def test_simple(self):
        gt_bboxes, pred_bboxes, gt_labels, pred_labels, pred_scores = \
            get_data()
        mean_ap: float = metrics.compute_mean_average_precision(
            gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores
        )

        self.assertAlmostEqual(mean_ap, 1.0)
        self.assertIsInstance(mean_ap, float)

    def test_map_is_not_one(self):
        gt_bboxes, pred_bboxes, gt_labels, pred_labels, pred_scores = \
            get_data()
        gt_bboxes[0] = torch.Tensor([20, 20, 30, 40])

        mean_ap: float = metrics.compute_mean_average_precision(
            gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores
        )
        self.assertAlmostEqual(mean_ap, 0.5)
