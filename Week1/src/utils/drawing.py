from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import draw_bounding_boxes

from .conversion import bbox_conversion


def draw_bbox(
    image: np.ndarray | torch.Tensor | Image.Image,
    bbox: dict,
    id2label: dict,
    format: Literal["coco", "pascal_voc", "yolo"] = "pascal_voc",
    boxes_key: str = "boxes",
    labels_key: str = "labels",
    scores_key: str = "scores",
):
    boxes = bbox[boxes_key]
    if len(boxes) == 0:
        return image

    # Parse image
    is_tensor = isinstance(image, torch.Tensor)
    is_pil = isinstance(image, Image.Image)
    is_array = isinstance(image, np.ndarray)
    if not is_tensor:
        image = ToTensor()(image)

    boxes = bbox_conversion(
        bbox, iformat=format, oformat="pascal_voc", image_size=image.shape[-2:]
    )
    boxes = torch.as_tensor(boxes)
    labels = bbox[labels_key]
    scores = bbox[scores_key]

    names = [
        f"{id2label[label.item()]}: {score:.2f}" for label, score in zip(labels, scores)
    ]

    drawing = draw_bounding_boxes(image, boxes, labels=names, colors="yellow", width=3)

    if is_pil:
        drawing = ToPILImage()(drawing)
    if is_array:
        drawing = ToPILImage()(drawing)
        drawing = np.array(drawing)
        drawing = drawing

    return drawing


if __name__ == "__main__":
    pass
