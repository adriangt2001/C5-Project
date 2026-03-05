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
        mode: Literal["train", "test"] = "test",
        transforms=None,
    ):
        self.transforms = transforms
        self.mode = mode
        self.features = {
            "image_id": [],
            "image": [],
            "width": [],
            "height": [],
            "objects": [],
        }
        seqmap, max_frames = motsio.load_seqmap(seqmap_file)

        for seq in seqmap:
            self.load_sequence(
                os.path.join(annotations_folder, "image_02", f"{seq}.txt"),
                os.path.join(image_folder, "image_02", f"{seq}.txt"),
            )

    def load_sequence(self, txt_path, image_folder):
        loaded_txt = motsio.load_txt(txt_path)

        features = {"image": [], "width": [], "height": [], "objects": []}

        for image in sorted(glob.glob(f"{image_folder}/*.png")):
            img = Image.open(image)
            features["image"].append(img)
            features["width"].append(img.size[0])
            features["height"].append(img.size[1])
            if self.mode == "train":
                features["objects"].append(
                    {"id": [], "area": [], "mask": [], "category": []}
                )

        if self.mode == "train":
            for frame_idx, frame_info in loaded_txt.items():
                for segmentation in frame_info:
                    if segmentation.class_id in [1, 2]:
                        features["objects"][frame_idx]["area"].append(
                            rletools.area(segmentation.mask)
                        )
                        features["objects"][frame_idx]["mask"].append(
                            rletools.toBbox(segmentation.mask)
                        )
                        features["objects"][frame_idx]["category"].append(
                            segmentation.class_id
                        )

        self.features["image"].extend(features["image"])
        self.features["width"].extend(features["width"])
        self.features["height"].extend(features["height"])
        self.features["objects"].extend(features["objects"])

    def __getitem__(self, idx):
        # This is untested.
        # Be sure that the segmentation mask returns an empty list for the "test" mode,
        # and a list of masks (?) when in "train" mode,
        image = np.array(self.features["image"][idx])
        # Mask will be empty if test mode is set
        mask = self.features["objects"][idx]["mask"]

        if self.transforms:
            # This format is for albumentations when using semantic segmentation
            # (perhaps it needs to be changed for instance segmentation)
            output = self.transforms(image=image, mask=mask)

        return output["image"], output["mask"]
