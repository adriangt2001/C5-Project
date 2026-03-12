from __future__ import annotations

from typing import Dict, List

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class InstanceSegmentationMetrics:
    def __init__(self, class_metrics: bool = True):
        self.metric = MeanAveragePrecision(
            iou_type="segm",
            class_metrics=class_metrics,
        )

    def update(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        self.metric.update(preds, targets)

    def compute(self) -> Dict[str, torch.Tensor]:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

def parse_segmentation_metrics(
    results: Dict[str, torch.Tensor],
    id2label: dict,
) -> Dict[str, float]:
    metrics = {}

    for k, v in results.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            metrics[k] = round(v.item(), 4)

    if "classes" in results and "map_per_class" in results and "mar_100_per_class" in results:
        classes = results["classes"]
        map_per_class = results["map_per_class"]
        mar_per_class = results["mar_100_per_class"]

        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_per_class):
            class_id = int(class_id.item())
            class_name = id2label[class_id]
            metrics[f"map_{class_name}"] = round(class_map.item(), 4)
            metrics[f"mar_100_{class_name}"] = round(class_mar.item(), 4)

    return metrics

class SemanticSegmentationMetrics:
    def __init__(self, num_classes=3, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def reset(self):
        self.confusion_matrix.zero_()

    def update(self, pred, target):
        """
        pred:   [H, W] predicted semantic mask
        target: [H, W] ground truth semantic mask
        """

        pred = pred.view(-1)
        target = target.view(-1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]

        valid = (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]

        indices = self.num_classes * target + pred
        cm = torch.bincount(indices, minlength=self.num_classes ** 2)
        cm = cm.reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += cm.cpu()

    def compute(self):
        cm = self.confusion_matrix.float()

        tp = torch.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        iou = tp / (tp + fp + fn + 1e-8)
        pixel_accuracy = tp.sum() / (cm.sum() + 1e-8)
        mean_accuracy = (tp / (tp + fn + 1e-8)).mean()

        return {
            "pixel_accuracy": pixel_accuracy.item(),
            "mean_accuracy": mean_accuracy.item(),
            "iou_per_class": iou,
            "miou": iou.mean().item(),
            "confusion_matrix": cm
        }