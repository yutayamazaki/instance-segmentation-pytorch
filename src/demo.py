import argparse
import logging.config
import os
from logging import getLogger
from typing import Any, Callable, Dict, List, Tuple

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

import utils
import visualize as vis
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
    logger.info(f'Detected {len(out["masks"])} objects.')

    # remove bounding box by confidence threshold.
    removed: Dict[str, Any] = {
        'boxes': [], 'labels': [], 'scores': [], 'masks': []
    }
    score_threshold: float = 0.9
    for box_idx, score in enumerate(out['scores']):
        if score >= score_threshold:
            removed['boxes'].append(out['boxes'][box_idx].tolist())
            removed['scores'].append(float(out['scores'][box_idx]))
            removed['masks'].append(out['masks'][box_idx])
    out = removed

    # Draw boxes and its masks.
    # [x0, y0, x1, y1]
    boxes: List[List[float]] = out['boxes']
    labels: List[str] = ['gray' for _ in boxes]
    scores: List[float] = removed['scores']
    img_boxes = vis.draw_boxes(orig_img, boxes, labels, scores)

    img_arr = np.array(orig_img)
    masks: List[np.ndarray] = []
    for mask in out['masks']:
        _, h, w = mask.size()
        mask_arr: np.ndarray = mask.reshape(h, w).detach().numpy()
        masks.append(mask_arr)

    colors: List[Tuple[int, ...]] = [
        vis.random_cmap() for _ in range(len(masks))
    ]
    img_mask = vis.apply_masks(img_arr, masks, colors)
    img_mask = Image.fromarray(np.uint8(img_mask))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plt.grid(None)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.grid(None)

    ax1.imshow(Image.open(args.img_path).convert('RGB'))
    ax1.set_title('original')
    ax2.imshow(img_mask)
    ax2.set_title('prediction')

    file_id, _ = os.path.splitext(os.path.basename(args.img_path))
    plt.savefig(f'../visualized_results/{file_id}.png')
