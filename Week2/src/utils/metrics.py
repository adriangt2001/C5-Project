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