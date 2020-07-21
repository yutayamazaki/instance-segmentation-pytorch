import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def draw_boxes(
    img: Image.Image, boxes: List[List[float]],
    labels: List[str], scores: List[float]
) -> Image.Image:
    """Draw bouding boxes to given Pillow Image.
    Args:
        img (Image.Image): Image to draw.
        boxes (List[List[float]]): Like [[xmin, ymin, xmax, ymax], ...].
        labels (List[str]): A list of labels of boxes.
        scores (List[float]): A list of prediction confidences.

    Return:
        (Image.Image): An image with bounding boxes.
    """
    assert len(boxes) == len(labels) == len(scores)

    label2cmap: Dict[str, Tuple[int, ...]] = {
        label: random_cmap() for label in set(labels)
    }

    draw = ImageDraw.Draw(img)
    # [x0, y0, x1, y1]

    for box, label, score in zip(boxes, labels, scores):
        x_min: float = max(0, box[0])
        y_min: float = max(0, box[1] - 12)
        draw.rectangle(xy=box, outline=label2cmap[label])
        text: str = f'{label}: {score:.2f}'
        draw.text(
            xy=(x_min, y_min), text=text, fill=(255, 255, 255)
        )
    return img


def random_cmap() -> Tuple[int, ...]:
    """Generate random color map. Example return is (r, g, b)."""
    # Include 0 and 255.
    # return tuple([random.randint(100, 255), random.randint(0, 255), 0])
    return tuple(random.randint(0, 255) for _ in range(3))


def random_cmaps(n: int) -> List[Tuple[int, ...]]:
    """Create specified number of color maps."""
    return [random_cmap() for _ in range(n)]


def apply_mask(
    img: np.ndarray, mask: np.ndarray, color: Tuple[int, ...],
    alpha: float = 0.5, threhold: float = 0.9
) -> np.ndarray:
    """Apply mask to given image.
    Args:
        img (np.ndarray): (Height, Width, Channel).
        mask (np.ndarray): A shape of mask is (Hegith, Width).
        color (Tuple[int, ...]): Like (r, g, b).
        alpha (float): Weights of given color.
                       Each pixel value is (1-alpha) * img + alpha * color.
        threshold (float): Mask binarize threshold.

    Returns:
        np.ndarray: Masked image with shape (Height, Width, Channel).
    """
    img_copy: np.ndarray = np.copy(img)
    """Apply the given mask to the image."""
    for c_idx in range(3):
        img_copy[:, :, c_idx] = np.where(
            mask > threhold,
            img_copy[:, :, c_idx] * (1 - alpha) + alpha * color[c_idx],
            img_copy[:, :, c_idx]
        )
    return img_copy


def apply_masks(
    img: np.ndarray, masks: List[np.ndarray], colors: List[Tuple[int, ...]],
    alpha: float = 0.5, threhold: float = 0.9
) -> np.ndarray:
    """Apply masks to given image.
    Args:
        img (np.ndarray): (Height, Width, Channel).
        masks (List[np.ndarray]): Shape of each mask is (Hegith, Width).
        colors (List[Tuple[int, ...]]): Like [(r, g, b), ...].
        alpha (float): Weights of given color.
                       Each pixel value is (1-alpha) * img + alpha * color.
        threshold (float): Mask binarize threshold.

    Returns:
        np.ndarray: Masked image with shape (Height, Width, Channel).
    """
    assert img.ndim == 3
    assert len(masks) == len(colors)

    img_copy: np.ndarray = np.copy(img)
    for mask, color in zip(masks, colors):
        img_copy = apply_mask(img_copy, mask, color, alpha, threhold)

    return img_copy
