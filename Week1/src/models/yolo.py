import os
from ultralytics import YOLO
from PIL import Image
from typing import List
import torch
import torch.nn.functional as F
import math
import logging


# Set logging level to WARNING to suppress INFO logs
logging.getLogger('ultralytics').setLevel(logging.WARNING)


class YOLOModel:

    def __init__(self, model: str = "yolov10m.pt", device=None):
        self.device = device
        # load model
        self.model = YOLO(os.path.join("models", model), verbose=False)
        self.model.to(self.device)

    def _pad_to_shape(self, x: torch.Tensor, target_h: int, target_w: int, value: float = 0.5):
        """
        Pad a single image tensor (C,H,W) to (C,target_h,target_w) with symmetric padding.
        Returns padded tensor and (left, top) padding offsets.
        """
        assert x.ndim == 3, f"Expected (C,H,W), got {x.shape}"
        c, h, w = x.shape
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"Target shape smaller than input: input {(h,w)} target {(target_h,target_w)}")

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        x = F.pad(x, (left, right, top, bottom), value=value)
        return x, (left, top)

    def _make_batch(self, imgs: List[torch.Tensor], stride: int = 32, pad_value: float = 0.5):
        """
        imgs: list of (C,H,W) tensors (can vary H,W)
        Returns:
          batch: (B,C,Hmax,Wmax) float32 in [0,1]
          pads: list of (pad_left, pad_top) per image
          orig_sizes: list of (h,w) per image
        """
        # normalize each image first to float32 [0,1]
        norm = [self._normalize_images(im) for im in imgs]

        # record original sizes
        orig_sizes = [(int(im.shape[1]), int(im.shape[2])) for im in norm]  # (H,W)

        # choose a common target H,W for the batch: max dims, rounded to stride
        max_h = max(h for h, w in orig_sizes)
        max_w = max(w for h, w in orig_sizes)
        target_h = math.ceil(max_h / stride) * stride
        target_w = math.ceil(max_w / stride) * stride

        padded = []
        pads = []
        for im in norm:
            im_pad, (pl, pt) = self._pad_to_shape(im, target_h, target_w, value=pad_value)
            padded.append(im_pad)
            pads.append((pl, pt))

        batch = torch.stack(padded, dim=0)  # (B,C,H,W)
        return batch, pads, orig_sizes

    def _normalize_images(self, x):
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return x.clamp(0.0, 1.0)
    

    def inference(self, img):
        pads = None
        orig_sizes = None

        if isinstance(img, list) and len(img) > 0 and isinstance(img[0], torch.Tensor):
            # expect list of (C,H,W), possibly different sizes
            img, pads, orig_sizes = self._make_batch(img, stride=32, pad_value=0.5)

        results = self.model(img)

        predictions = []
        for i, result in enumerate(results):
            pred = {
                "boxes": result.boxes.xyxy,        # (N,4) in padded image coords
                "labels": result.boxes.cls.int(),  # (N,)
                "scores": result.boxes.conf        # (N,)
            }

            # Convert boxes back in ORIGINAL (unpadded) coords:
            if pads is not None and orig_sizes is not None:
                pad_left, pad_top = pads[i]
                h0, w0 = orig_sizes[i]
                boxes = pred["boxes"].clone()
                boxes[:, [0, 2]] -= pad_left
                boxes[:, [1, 3]] -= pad_top
                # clip to original image bounds
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w0)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h0)
                pred["boxes"] = boxes

            predictions.append(pred)

        return predictions

    def evaluate(self, images, targets, metric):
        """
        Processes a batch for evaluation, applying the class mapping and 
        updating the provided torchmetrics MeanAveragePrecision object.
        
        Args:
            images: Batch of images from DataLoader.
            targets: Batch of ground truth dictionaries from DataLoader.
            metric: torchmetrics.detection.mean_ap.MeanAveragePrecision instance.
        """
        preds = self.inference(images)

        person_key = next(k for k, v in self.model.names.items() if v == "person")
        car_key = next(k for k, v in self.model.names.items() if v == "car")
        kitti_mapping = {
            car_key: 1,
            person_key: 2
        }

        processed_preds = []
        for p in preds:
            # Keep only classes present in our mapping (Pedestrian and Car)
            keep = [i for i, label in enumerate(p['labels']) if label.item()
                    in kitti_mapping]

            processed_preds.append({
                'boxes': p['boxes'][keep].cpu(),
                'scores': p['scores'][keep].cpu(),
                'labels': torch.tensor(
                    [kitti_mapping[l.item()] for l in p['labels'][keep]],
                    dtype=torch.int64)
            })

        targets_cpu = [{k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]
        metric.update(processed_preds, targets_cpu)



    def train(self):
        pass
