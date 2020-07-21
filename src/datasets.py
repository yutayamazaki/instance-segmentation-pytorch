import glob
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image


def load_images_and_masks(
    root: str = '../datumaro'
) -> Tuple[List[str], List[str]]:
    mask_paths: List[str] = sorted(
        glob.glob(os.path.join(root, 'masks', '*.png'))
    )

    ret_masks: List[str] = []
    ret_images: List[str] = []
    for mask_path in mask_paths:
        fname: str = os.path.basename(mask_path)
        file_id, _ = os.path.splitext(fname)
        img_path: str = os.path.join(root, f'{file_id}.jpg')

        if os.path.exists(img_path):
            ret_images.append(img_path)
            ret_masks.append(mask_path)

    return ret_images, ret_masks


class InstanceSegmentationDataset(torch.utils.data.Dataset):

    def __init__(self, transforms: Callable):
        self.transforms: Callable = transforms

        img_paths, mask_paths = load_images_and_masks()
        self.img_paths: List[str] = img_paths
        self.mask_paths: List[str] = mask_paths

    def __getitem__(self, idx):
        pil_img = Image.open(self.img_paths[idx]).convert('RGB')
        pil_mask = Image.open(self.mask_paths[idx])
        img_arr: np.array = np.array(pil_img)
        mask_arr: np.array = np.array(pil_mask)

        aug: Dict[str, np.ndarray] = self.transforms(
            image=img_arr, mask=mask_arr
        )
        img: torch.Tensor = torch.as_tensor(
            aug['image'].transpose((2, 0, 1))
        ).float()
        mask: np.ndarray = aug['mask']

        obj_ids: np.ndarray = np.unique(mask)
        obj_ids = obj_ids[1:]  # remove background.

        masks = mask == obj_ids[:, None, None]

        num_objs: int = len(obj_ids)
        boxes: List[List[float]] = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes: torch.Tensor = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels: torch.Tensor = torch.ones((num_objs,), dtype=torch.int64)
        masks: torch.Tensor = torch.as_tensor(masks, dtype=torch.uint8)

        image_id: torch.Tensor = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd: torch.Tensor = torch.zeros((num_objs,), dtype=torch.int64)

        target: Dict[torch.Tensor] = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        return img, target

    def __len__(self) -> int:
        return len(self.img_paths)


def load_masks(path: str):
    pil_mask = Image.open(path)
    mask_arr: np.ndarray = np.array(pil_mask)

    obj_ids: np.ndarray = np.unique(mask_arr)
    obj_ids = obj_ids[1:]  # remove background.

    masks = mask_arr == obj_ids[:, None, None]

    num_objs: int = len(obj_ids)
    boxes: List[List[float]] = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)  # type: ignore
    # there is only one class
    labels: torch.Tensor = torch.ones((num_objs,), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    return boxes, labels, masks
