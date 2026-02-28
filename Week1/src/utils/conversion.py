from typing import Literal

import numpy as np
import torch

from albumentations.core.bbox_utils import (
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
)


def bbox_conversion(
    bbox: torch.Tensor | np.ndarray,
    iformat: Literal["yolo", "pascal_voc", "coco"],
    oformat: Literal["yolo", "pascal_voc", "coco"],
    image_size: tuple[int, int],
):
    is_tensor = isinstance(bbox, torch.Tensor)
    if is_tensor:
        bbox = bbox.numpy()

    shape = {"height": image_size[0], "width": image_size[1]}

    bbox = convert_bboxes_to_albumentations(bbox, iformat, shape)
    bbox = convert_bboxes_from_albumentations(bbox, oformat, shape)

    if is_tensor:
        bbox = torch.as_tensor(bbox)

    return bbox


if __name__ == "__main__":
    from itertools import product

    from tqdm import tqdm

    formats = ["coco", "yolo", "pascal_voc"]
    image_size = [128, 512]
    t = torch.randint(1, 255, [3, *image_size])
    bboxes = [t]

    # Test bbox_conversion shape
    combos = product(bboxes, formats, formats)
    for bbox, iformat, oformat in tqdm(combos):
        expected_shape = bbox.shape
        output_bbox = bbox_conversion(
            bbox, iformat=iformat, oformat=oformat, image_size=(512, 512)
        )
        assert expected_shape == output_bbox.shape, (
            f"Output shape: {output_bbox.shape}\nExpected shape: {expected_shape}"
        )
