import numpy as np
import torch
from src.utils.conversion import bbox_conversion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoImageProcessor, EvalPrediction


class ModelOutput:
    @torch.no_grad()
    def __init__(self, logits: torch.Tensor, pred_boxes: torch.Tensor):
        self.logits: torch.Tensor = logits
        self.pred_boxes: torch.Tensor = pred_boxes


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    id2label: dict,
    threshold: float = 0.5,
):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []
    target_classes = torch.tensor(list(id2label.keys()))

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = bbox_conversion(
                image_target["boxes"], "yolo", "pascal_voc", image_target["orig_size"]
            )
            boxes = torch.as_tensor(boxes)
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(
            logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
        )
        post_processed_output = image_processor.post_process_object_detection(
            output,
            threshold=threshold,
            target_sizes=target_sizes,
        )
        # Iterate through the predictions for each image in the batch
        for pred_dict in post_processed_output:
            # Create a mask where True means the label is either 1 (person) or 3 (car)
            mask = torch.isin(pred_dict["labels"], target_classes)

            # Apply the mask to boxes, scores, and labels to keep only classes 1 and 3
            filtered_pred = {
                "boxes": pred_dict["boxes"][mask],
                "scores": pred_dict["scores"][mask],
                "labels": pred_dict["labels"][mask],
            }
            post_processed_predictions.append(filtered_pred)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(
        classes, map_per_class, mar_100_per_class
    ):
        class_name = id2label[class_id.item()]
        metrics[f"map_class_{class_name.title()}"] = class_map
        metrics[f"mar_100_class_{class_name.title()}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (list[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (list[float]): list of corresponding areas to provided bounding boxes
        bboxes (list[tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
    examples, transform, image_processor, class_map, return_pixel_mask=False
):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        image = np.array(image.convert("RGB"))

        output = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )
        images.append(output["image"])

        # map from dataset to model labels
        output["category"] = [class_map[x] for x in output["category"]]

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(
        images=images, annotations=annotations, return_tensors="pt"
    )

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result
