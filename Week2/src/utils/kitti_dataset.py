import glob
import os
from typing import Literal

import numpy as np
import pycocotools.mask as rletools
import torch
from PIL import Image

from . import motsio

class KittiDataset(torch.utils.data.Dataset):

    def __init__(self, image_folder, annotations_folder):

        self.image_folder = image_folder
        self.annotations_folder = annotations_folder

        self.samples = []

        annotation_files = sorted(glob.glob(os.path.join(annotations_folder, "*.txt")))

        for ann_file in annotation_files:

            seq_id = os.path.basename(ann_file).split(".")[0]

            image_seq_folder = os.path.join(image_folder, seq_id)

            images = sorted(glob.glob(os.path.join(image_seq_folder, "*.png")))

            for frame_id, img_path in enumerate(images):

                self.samples.append((img_path, ann_file, frame_id))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, ann_file, frame_id = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        height, width = image.shape[:2]

        masks = []

        with open(ann_file, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            frame = int(parts[0])

            if frame != frame_id:
                continue

            class_id = int(parts[2])

            # skip ignore regions
            if class_id == 10:
                continue

            height = int(parts[3])
            width = int(parts[4])
            rle_string = parts[5]

            rle = {
                "size": [height, width],
                "counts": rle_string.encode("utf-8")
            }

            mask = rletools.decode(rle)

            masks.append(mask)

        if len(masks) > 0:
            masks = np.stack(masks)
        else:
            masks = np.zeros((0, height, width))

        return {
            "image": torch.tensor(image).permute(2,0,1).float(),
            "masks": torch.tensor(masks).float(),
            "class_ids": torch.tensor([int(line.strip().split()[2]) for line in lines if int(line.strip().split()[0]) == frame_id and int(line.strip().split()[2]) != 10]).long()
        }