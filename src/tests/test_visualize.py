import unittest
from typing import List, Tuple

import numpy as np
from PIL import Image

import visualize as vis


class DrawBoxesTests(unittest.TestCase):

    def test_simple(self):
        img: Image.Image = Image.fromarray(np.uint8(np.zeros((8, 16, 3))))
        boxes: List[List[float]] = [[0., 1., 2., 4.], [2., 3., 4., 5.]]
        labels: List[str] = ['person', 'dog']
        scores: List[float] = [0.9, 0.3]
        ret: Image.Image = vis.draw_boxes(img, boxes, labels, scores)
        self.assertIsInstance(ret, Image.Image)
        self.assertEqual(ret.height, 8)
        self.assertEqual(ret.width, 16)


class RanomCmapTests(unittest.TestCase):

    def test_simple(self):
        cmap: Tuple[int, ...] = vis.random_cmap()
        self.assertIsInstance(cmap, tuple)
        self.assertEqual(len(cmap), 3)
        self.assertTrue(max(cmap) <= 255)
        self.assertTrue(min(cmap) >= 0)


class RanomCmapsTests(unittest.TestCase):

    def test_simple(self):
        cmaps: List[Tuple[int, ...]] = vis.random_cmaps(n=2)
        self.assertIsInstance(cmaps, list)
        self.assertEqual(len(cmaps), 2)


class ApplyMaskTests(unittest.TestCase):

    def test_simple(self):
        img: np.ndarray = np.zeros((24, 16, 3))
        mask: np.ndarray = np.zeros((24, 16))
        mask[0, 0] = 1.
        color: Tuple[int, ...] = (128, 128, 128)
        ret: np.ndaray = vis.apply_mask(img, mask, color)

        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret.shape, (24, 16, 3))
        np.testing.assert_array_equal(np.array([64., 64., 64.]), ret[0, 0, :])


class ApplyMasksTests(unittest.TestCase):

    def test_simple(self):
        img: np.ndarray = np.zeros((24, 16, 3))
        masks: List[np.ndarray] = [np.zeros((24, 16)), np.zeros((24, 16))]
        masks[0][0, 0] = 1.
        color: List[Tuple[int, ...]] = [
            (128, 128, 128) for _ in range(len(masks))
        ]
        ret: np.ndaray = vis.apply_masks(img, masks, color)

        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret.shape, (24, 16, 3))
        np.testing.assert_array_equal(np.array([64., 64., 64.]), ret[0, 0, :])
