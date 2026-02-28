from typing import Literal

import numpy as np
import torch


def bbox_conversion(
    bbox: torch.Tensor | np.ndarray,
    iformat: Literal["xyxy", "xywh", "cxcywh"],
    oformat: Literal["xyxy", "xywh", "cxcywh"],
    image_size: tuple[int, int],
):
    if iformat == oformat:
        return bbox

    if (iformat == "cxcywh" or oformat == "cxcywh") and image_size is None:
        raise ValueError(
            "Need an image to translate to the 'cxcywh' (relative values) format."
        )

    if image_size is not None:
        image_size = np.array(image_size, dtype=float)

    is_tensor = isinstance(bbox, torch.Tensor)
    if is_tensor:
        bbox = bbox.numpy()

    if iformat == "xyxy":
        x = bbox[..., 0]
        y = bbox[..., 1]
        w = bbox[..., 2] - bbox[..., 0]
        h = bbox[..., 3] - bbox[..., 1]
        if oformat == "xywh":
            new_bbox = np.stack([x, y, w, h], axis=-1)
        elif oformat == "cxcywh":
            x = x / image_size[0]
            y = y / image_size[1]
            w = w / image_size[0]
            h = h / image_size[1]
            cx = x + (x - w) / 2
            cy = y + (y - h) / 2
            new_bbox = np.stack([cx, cy, w, h], axis=-1)
    elif iformat == "xywh":
        x = bbox[..., 0]
        y = bbox[..., 1]
        w = bbox[..., 2]
        h = bbox[..., 3]
        if oformat == "xyxy":
            x_max = x + w
            y_max = y + h
            new_bbox = np.stack([x, y, x_max, y_max], axis=-1)
        elif oformat == "cxcywh":
            x = x / image_size[0]
            y = y / image_size[1]
            w = w / image_size[0]
            h = h / image_size[1]
            cx = x + (x - w) / 2
            cy = y + (y - h) / 2
            new_bbox = np.stack([cx, cy, w, h], axis=-1)
    elif iformat == "cxcywh":
        x = bbox[..., 0]
        y = bbox[..., 1]
        w = bbox[..., 2]
        h = bbox[..., 3]
        x_min = bbox[..., 0] - w / 2
        y_min = bbox[..., 1] - h / 2
        if oformat == "xyxy":
            x_max = x_min + w
            y_max = y_min + h
            new_bbox = np.stack([x_min, y_min, x_max, y_max], axis=-1)
        elif oformat == "xywh":
            new_bbox = np.stack([x_min, y_min, w, h], axis=-1)

    new_bbox = new_bbox.astype(int)
    if is_tensor:
        new_bbox = torch.from_numpy(new_bbox)

    return new_bbox


if __name__ == "__main__":
    from itertools import product

    from tqdm import tqdm

    formats = ["xyxy", "xywh", "cxcywh"]
    t = torch.randint(1, 255, [23, 18, 4])
    n = np.random.randint(1, 255, [67, 6, 4])
    bboxes = [t, n]

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
