import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class DEArtDatasetTorchvision(Dataset):
    """
    DEArt Dataset (Pascal VOC XML format) for torchvision detection models.

    Returns:
        image: Tensor [3, H, W]
        target: dict:
            - boxes: FloatTensor [N,4] (XYXY)
            - labels: Int64Tensor [N]
            - image_id: Tensor [1]
            - area: FloatTensor [N]
            - iscrowd: Int64Tensor [N]
    """

    def __init__(
        self,
        root,
        transform=None,
        split=None,               # "train", "val", "test", or None
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=42,
    ):
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.annots_dir = os.path.join(root, "annots_pub")
        self.transform = transform

        assert os.path.exists(self.images_dir), f"Images folder not found: {self.images_dir}"
        assert os.path.exists(self.annots_dir), f"Annotations folder not found: {self.annots_dir}"

        # -------------------------------------------------
        # 1️⃣ Collect valid samples (xml + image exists)
        # -------------------------------------------------
        all_xml = sorted(
            f for f in os.listdir(self.annots_dir) if f.endswith(".xml")
        )

        samples = []
        for xml_file in all_xml:
            xml_path = os.path.join(self.annots_dir, xml_file)
            tree = ET.parse(xml_path)
            root_xml = tree.getroot()

            filename = root_xml.find("filename").text
            img_path = os.path.join(self.images_dir, filename)

            if os.path.exists(img_path):
                samples.append((xml_file, filename))

        if len(samples) == 0:
            raise RuntimeError("No valid samples found.")

        # -------------------------------------------------
        # 2️⃣ Deterministic Train / Val / Test split
        # -------------------------------------------------
        torch.manual_seed(random_seed)
        indices = torch.randperm(len(samples)).tolist()

        n_total = len(samples)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        if split is None:
            selected = indices
        elif split == "train":
            selected = train_idx
        elif split == "val":
            selected = val_idx
        elif split == "test":
            selected = test_idx
        else:
            raise ValueError("split must be 'train', 'val', 'test' or None")

        self.samples = [samples[i] for i in selected]

        print(f"Total valid samples: {n_total}")
        print(f"Split '{split}' -> {len(self.samples)} samples")

        # -------------------------------------------------
        # 3️⃣ Build class mapping from ALL samples
        # -------------------------------------------------
        self.class_to_idx = self._build_class_mapping(samples)

    # -----------------------------------------------------
    # Build class mapping (background = 0)
    # -----------------------------------------------------
    def _build_class_mapping(self, samples):
        classes = set()

        for xml_file, _ in samples:
            xml_path = os.path.join(self.annots_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                classes.add(obj.find("name").text)

        classes = sorted(classes)
        return {cls_name: idx + 1 for idx, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------
    # Load one sample
    # -----------------------------------------------------
    def __getitem__(self, idx):

        xml_file, filename = self.samples[idx]

        xml_path = os.path.join(self.annots_dir, xml_file)
        img_path = os.path.join(self.images_dir, filename)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_pil = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            label = self.class_to_idx[name]

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # -------------------------------------------------
        # Apply Albumentations (if provided)
        # -------------------------------------------------
        if self.transform is not None:
            augmented = self.transform(
                image=np.array(image_pil),
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
            )

            image = augmented["image"]

            if len(augmented["bboxes"]) > 0:
                boxes = torch.tensor(augmented["bboxes"], dtype=torch.float32)
                labels = torch.tensor(augmented["labels"], dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        else:
            image = ToTensor()(image_pil)

        # -------------------------------------------------
        # Required fields for torchvision detection
        # -------------------------------------------------
        if boxes.shape[0] > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        return image, target