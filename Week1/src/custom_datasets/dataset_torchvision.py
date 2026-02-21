import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from .dataset_base import KittiDataset


class KittiDatasetTorchvision(KittiDataset):
    """
    Wrapper around the base KITTI dataset.

    Converts the output format to match what torchvision detection
    models (Faster R-CNN) expect:
        - boxes in XYXY format
        - labels as int64 tensors
        - additional required fields (area, iscrowd, image_id)
    """
    def __getitem__(self, idx):
        # Load image
        img_path = self.features["image"][idx]
        image = Image.open(img_path).convert("RGB")
        image = ToTensor()(image)  # Tensor [3, H, W] in range [0,1]

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

if __name__ == "__main__":
    ds = KittiDatasetTorchvision(root="../../dataset/KITTI-MOTS", annotations_folder="instances_txt", image_folder="training", seqmap_file="train.seqmap"
    )

    print("Dataset length:", len(ds))

    img, target = ds[0]

    print("Image shape:", img.shape)
    print("Target keys:", target.keys())
    print("Boxes shape:", target["boxes"].shape)
    print("Labels shape:", target["labels"].shape)
