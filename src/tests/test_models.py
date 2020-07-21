import unittest

import torchvision

import models


class GetInstanceSegmentationModelTests(unittest.TestCase):

    def test_simple(self):
        net = models.get_instance_segmentation_model(
            num_classes=2, pretrained=False
        )
        self.assertIsInstance(
            net, torchvision.models.detection.mask_rcnn.MaskRCNN
        )
