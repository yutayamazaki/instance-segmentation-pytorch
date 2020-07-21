import argparse
import logging.config
from logging import getLogger
from typing import Any, Callable, Dict, List, Tuple

import albumentations as albu
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

import metrics
import utils
from datasets import load_images_and_masks
from models import get_instance_segmentation_model


def load_inputs(img_path: str, transforms: Callable) -> torch.Tensor:
    orig_img: Image.Image = Image.open(img_path).convert('RGB')
    img: np.ndarray = transforms(image=np.array(orig_img))['image']
    img = torch.as_tensor(img).permute(2, 0, 1)
    c, h, w = img.size()
    img = img.reshape(1, c, h, w)
    return img


def load_mask_and_bboxes(mask_path: str) -> Tuple[np.ndarray, torch.Tensor]:
    pil_mask = Image.open(mask_path)
    mask: np.ndarray = np.array(pil_mask)
    obj_ids: np.ndarray = np.unique(mask)
    obj_ids = obj_ids[1:]

    masks: np.ndarray = mask == obj_ids[:, None, None]

    num_objs: int = len(obj_ids)
    boxes: List[List[float]] = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

    boxes_tensor: torch.Tensor = torch.as_tensor(boxes, dtype=torch.float32)
    return masks, boxes_tensor


if __name__ == '__main__':
    utils.seed_everything(seed=428)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights_path', type=str,
        default='../experiments/20200711_14-19-46/weights/weights.pth'
    )
    parser.add_argument(
        '-c', '--config_path', type=str,
        default='../experiments/20200711_14-19-46/config.yml'
    )
    args = parser.parse_args()

    with open('logger_conf.yaml', 'r') as f:
        log_config: Dict[str, Any] = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)

    logger = getLogger(__name__)
    cfg_dict: Dict[str, Any] = utils.load_yaml(args.config_path)
    cfg: utils.DotDict = utils.DotDict(cfg_dict)

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_instance_segmentation_model(
        cfg.num_classes, cfg.pretrained
    )
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.to(device)
    model.eval()

    scores: Dict[str, List[float]] = {
        'bbox_iou': [],
        'mask_iou': [],
        'mAP': []
    }
    image_paths, mask_paths = load_images_and_masks()
    image_paths, mask_paths = image_paths[-170:], mask_paths[-170:]
    total: int = len(image_paths)
    for img_path, mask_path in tqdm(
        zip(image_paths, mask_paths), total=total
    ):
        albu_cfg: Dict[str, Any] = cfg.albumentations.eval.todict()
        transforms: Callable = albu.core.serialization.from_dict(albu_cfg)
        img = load_inputs(img_path, transforms)
        gt_masks, gt_bboxes = load_mask_and_bboxes(mask_path)

        out: Dict[str, torch.Tensor] = model(img.to(device))[0]

        # evaluation
        gt_labels = [torch.ones(1) for _ in range(len(gt_bboxes))]
        mean_ap: float = metrics.compute_mean_average_precision(
            gt_bboxes, gt_labels, out['boxes'], out['labels'], out['scores']
        )
        scores['mAP'].append(mean_ap)

        gt_masks = torch.as_tensor(gt_masks)
        box_mean_iou: float = metrics.bboxes_mean_iou(
            gt_bboxes, gt_labels, out['boxes'], out['labels'], out['scores'],
            iou_threshold=0.5, score_threshold=0.0
        )
        mask_mean_iou: float = metrics.masks_mean_iou(
            gt_bboxes, gt_masks, gt_labels, out['boxes'],
            torch.as_tensor(out['masks']),
            out['labels'], out['scores'],
            iou_threshold=0.5, score_threshold=0.0
        )
        scores['bbox_iou'].append(box_mean_iou)
        scores['mask_iou'].append(mask_mean_iou)

    bbox_miou: float = sum(scores['bbox_iou']) / len(scores['bbox_iou'])
    mask_miou: float = sum(scores['mask_iou']) / len(scores['mask_iou'])
    mean_ap = sum(scores['mAP']) / len(scores['mAP'])
    print(f'Mean Average Precision: {mean_ap}')
    print(f'Mask mean IoU: {mask_miou}')
    print(f'BBox mean IoU: {bbox_miou}')
