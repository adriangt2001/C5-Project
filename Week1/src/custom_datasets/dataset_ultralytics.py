from PIL import Image
import torch

from .dataset_base import KittiDataset


class KittiDatasetUltralytics(KittiDataset):
    """
    Wrapper around the base KITTI dataset
    
    Returns the output image as a PIL which is more suitable for 
    Ultralytics expected input
    """
    def __getitem__(self, idx):
        # Load image
        img_path = self.features["image"][idx]
        image = Image.open(img_path).convert("RGB")

        # Retrieve ground-truth bounding boxes (stored as XYWH)
        bboxes_xywh = self.features["objects"][idx]["bbox"]
        labels = self.features["objects"][idx]["category"]
        areas = self.features["objects"][idx]["area"]

        # Convert bounding boxes from XYWH to XYXY
        if len(bboxes_xywh) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(
                [[x, y, x + w, y + h] for (x, y, w, h) in bboxes_xywh],
                dtype=torch.float32
            )
            labels = torch.tensor(labels, dtype=torch.int64)
            area = torch.tensor(areas, dtype=torch.float32)

        # Assume no crowd annotations
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,        # Required format: XYXY
            "labels": labels,      # Class IDs
            "image_id": idx,       # Unique image identifier
            "area": area,
            "iscrowd": iscrowd,
        }

        return image, target
