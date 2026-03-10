import glob
import os
from typing import Literal

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
    ):
        self.transforms = transforms
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

    def __getitem__(self, idx):
        img_path, frame_objects = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        masks = []
        labels = []
        
        for obj in frame_objects:

            # Keep only cars and pedestrians
            if obj.class_id not in [1, 2]:
                continue

            mask = rletools.decode(obj.mask).astype(np.uint8)

            masks.append(mask)
            labels.append(obj.class_id)

        if len(masks) > 0:
            masks = np.stack(masks)
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        target = {
            "masks": torch.tensor(masks, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "img_path": img_path,
            "image_id": idx,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target