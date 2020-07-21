import argparse
import logging.config
import os
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional

import albumentations as albu
import numpy as np
import seaborn as sns
import torch
from PIL import Image

import metrics
import utils
from datasets import load_masks
from models import get_instance_segmentation_model


if __name__ == '__main__':
    utils.seed_everything(seed=428)
    sns.set()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights_path', type=str,
        default='../experiments/20200708_06-27-03/weights/weights.pth'
    )
    parser.add_argument(
        '-c', '--config_path', type=str
    )
    parser.add_argument(
        '-i', '--img_path', type=str
    )
    args = parser.parse_args()

    log_config: Dict[str, Any] = utils.load_yaml('logger_conf.yaml')
    logging.config.dictConfig(log_config)
    logger = getLogger(__name__)

    cfg_dict: Dict[str, Any] = utils.load_yaml(args.config_path)
    cfg: utils.DotDict = utils.DotDict(cfg_dict)

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and weights.
    net = get_instance_segmentation_model(
        num_classes=cfg.num_classes, pretrained=False
    )
    net.load_state_dict(torch.load(args.weights_path, map_location=device))

    # Prepare input image.
    orig_img: Image.Image = Image.open(args.img_path).convert('RGB')
    albu_cfg: Dict[str, Any] = cfg.albumentations.eval.todict()
    transforms: Callable = albu.core.serialization.from_dict(albu_cfg)
    img = transforms(image=np.array(orig_img))['image']
    img = torch.as_tensor(img).permute(2, 0, 1)
    c, h, w = img.size()
    img = img.reshape(1, c, h, w)

    # Predictioon.
    net.eval()
    out = net(img)[0]

    file_id, _ = os.path.splitext(os.path.basename(args.img_path))
    mask_path: str = os.path.join('../datumaro/masks', f'{file_id}.png')
    gt_boxes, gt_labels, gt_masks = load_masks(mask_path)

    mean_ap: float = metrics.compute_mean_average_precision(
        gt_boxes, gt_labels, out['boxes'], out['labels'], out['scores'], 0.5
    )
    print(mean_ap)
    exit()

    indices: List[Optional[int]] = metrics.box_matching(
        gt_boxes, gt_labels, gt_masks, out['boxes'], out['labels'],
        out['masks'], out['scores'], iou_threshold=0.5, score_threshold=0.6
    )

    gt_masks_ = []  # (N, H, W)
    gt_boxes_ = []  # (N, 4)
    pred_masks_ = []  # (N, 1, H, W)
    pred_boxes_ = []  # (N, 4)
    for i, idx in enumerate(indices):
        if idx is None:
            continue
        gt_masks_.append(gt_masks[i])
        gt_boxes_.append(gt_boxes[i])
        pred_masks_.append(out['masks'][idx])
        pred_boxes_.append(out['boxes'][idx])

    gt_masks_ = torch.stack(gt_masks_)
    pred_masks_ = torch.stack(pred_masks_)

    if len(gt_masks_) == 0:
        print('No matching prediction.')

    mask_mean_iou: float = metrics.mask_mean_iou(gt_masks_, pred_masks_)
    box_mean_iou: float = metrics.boxes_mean_iou(gt_boxes_, pred_boxes_)
    print(f'Mask mIoU: {mask_mean_iou}')
    print(f'Box mIoU: {box_mean_iou}')
