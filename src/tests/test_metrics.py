import unittest
from typing import List

import torch

import metrics


def get_data():
    gt_bboxes: List[torch.Tensor] = [
        torch.Tensor([0, 10, 10, 20]),
        torch.Tensor([0, 10, 10, 20]),
    ]
    pred_bboxes: List[torch.Tensor] = [
        torch.Tensor([0, 10, 10, 20]),
        torch.Tensor([0, 10, 10, 20]),
    ]
    gt_masks: torch.Tensor = torch.zeros((2, 30, 30))
    pred_masks: torch.Tensor = torch.zeros((2, 1, 30, 30))
    gt_labels: torch.Tensor = torch.ones(2)
    pred_labels: torch.Tensor = torch.ones(2)
    pred_scores: torch.Tensor = torch.Tensor([0.9, 0.8])
    return (
        gt_bboxes, pred_bboxes, gt_masks, pred_masks, gt_labels, pred_labels,
        pred_scores
    )


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


class MasksMeanIoUTests(unittest.TestCase):

    def test_simple(self):
        gt_bboxes, pred_bboxes, gt_masks, pred_masks, gt_labels, \
            pred_labels, pred_scores = get_data()
        gt_masks[0] = 1
        pred_masks[0] = 1
        mean_iou: float = metrics.masks_mean_iou(
            gt_bboxes, gt_masks, gt_labels, pred_bboxes, pred_masks,
            pred_labels, pred_scores
        )

        self.assertIsInstance(mean_iou, float)
        self.assertAlmostEqual(mean_iou, 0.5)

    def test_iou_zeros(self):
        gt_bboxes, pred_bboxes, gt_masks, pred_masks, gt_labels, \
            pred_labels, pred_scores = get_data()
        mean_iou: float = metrics.masks_mean_iou(
            gt_bboxes, gt_masks, gt_labels, pred_bboxes, pred_masks,
            pred_labels, pred_scores
        )

        self.assertAlmostEqual(mean_iou, 0.0)


class BBoxIoUTests(unittest.TestCase):

    def test_same_bbox_equal_to_one(self):
        # (xmin, ymin, xmax, ymax)
        bbox: torch.Tensor = torch.Tensor([0, 10, 10, 20])
        iou: float = metrics.bbox_iou(bbox, bbox)

        self.assertIsInstance(iou, float)
        self.assertAlmostEqual(iou, 1.)

    def test_iou_zero(self):
        # (xmin, ymin, xmax, ymax)
        bbox_a: torch.Tensor = torch.Tensor([0, 10, 10, 20])
        bbox_b: torch.Tensor = torch.Tensor([10, 20, 20, 25])
        iou: float = metrics.bbox_iou(bbox_a, bbox_b)

        self.assertAlmostEqual(iou, 0.)


class BBoxesMeanIoUTests(unittest.TestCase):

    def test_simple(self):
        gt_bboxes, pred_bboxes, _, _, gt_labels, pred_labels, pred_scores = \
            get_data()
        iou: float = metrics.bboxes_mean_iou(
            gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores,
            iou_threshold=0.5, score_threshold=0.0
        )
        self.assertIsInstance(iou, float)
        self.assertAlmostEqual(iou, 1.0)


class ComputeMeanAveragePrecisionTests(unittest.TestCase):

    def test_simple(self):
        gt_bboxes, pred_bboxes, _, _, gt_labels, pred_labels, pred_scores = \
            get_data()
        mean_ap: float = metrics.compute_mean_average_precision(
            gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores
        )

        self.assertAlmostEqual(mean_ap, 1.0)
        self.assertIsInstance(mean_ap, float)

    def test_map_is_not_one(self):
        gt_bboxes, pred_bboxes, _, _, gt_labels, pred_labels, pred_scores = \
            get_data()
        gt_bboxes[0] = torch.Tensor([20, 20, 30, 40])

        mean_ap: float = metrics.compute_mean_average_precision(
            gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores
        )
        self.assertAlmostEqual(mean_ap, 0.5)
