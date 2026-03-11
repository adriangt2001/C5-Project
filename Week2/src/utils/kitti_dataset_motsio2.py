import glob
import os
import numpy as np
import pycocotools.mask as rletools
import torch
from PIL import Image

from . import motsio


class KittiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_folder: str,
        annotations_folder: str,
        seqmap_file: str,
        transforms=None,
        only_mask: bool = False,
    ):
        self.transforms = transforms
        self.only_mask = only_mask
        self.samples = []

        seqmap, _ = motsio.load_seqmap(seqmap_file)

        for seq in seqmap:
            seq_image_folder = os.path.join(image_folder, "image_02", seq)
            seq_txt_path = os.path.join(annotations_folder, f"{seq}.txt")
            objects_per_frame = motsio.load_txt(seq_txt_path)
            image_paths = sorted(glob.glob(os.path.join(seq_image_folder, "*.png")))

            for frame_id, img_path in enumerate(image_paths):
                frame_objects = objects_per_frame.get(frame_id, [])
                self.samples.append((img_path, frame_objects))

    def __len__(self):
        return len(self.samples)

    def _mask_to_bbox(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()
        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        img_path, frame_objects = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        masks = []
        boxes = []
        labels = []

        for obj in frame_objects:

            # Keep only cars and pedestrians
            if obj.class_id not in [1, 2]:
                continue

            mask = rletools.decode(obj.mask).astype(np.uint8)

            bbox = self._mask_to_bbox(mask)
            if bbox is None:
                continue

            masks.append(mask)
            boxes.append(bbox)
            labels.append(obj.class_id)

        if len(masks) > 0:
            masks = np.stack(masks)
            boxes = np.array(boxes)
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)
            boxes = np.zeros((0, 4), dtype=np.float32)

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.only_mask:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        else:
            target["masks"] = torch.tensor(masks, dtype=torch.uint8)

        if self.transforms:
            image = self.transforms(image)

        return image, target