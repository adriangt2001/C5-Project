import numpy as np
import torch
import wandb
from src.utils.conversion import bbox_conversion
from src.utils.drawing import draw_bbox
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor,
    EvalPrediction,
    TrainerCallback,
    AutoModelForObjectDetection,
)
from peft import PeftModel


def load_model(model_variant: str, lora_path: str = None, merged: bool = True):
    model = AutoModelForObjectDetection.from_pretrained(model_variant)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        if merged:
            model = model.merge_and_unload()
    return model


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
    examples, image_processor, class_map, transform=None, return_pixel_mask=False
):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        image = np.array(image.convert("RGB"))

        if transform is not None:
            output = transform(
                image=image, bboxes=objects["bbox"], category=objects["category"]
            )
        else:
            output = {
                "image": image,
                "bboxes": objects["bbox"],
                "category": objects["category"],
            }
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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class WandbImageLoggerCallback(TrainerCallback):
    def __init__(self, num_samples, threshold, id2label, denormalize = True):
        self.num_samples = num_samples
        self.threshold = threshold
        self.id2label = id2label
        self.denormalize = denormalize

    def on_evaluate(
        self,
        args,
        state,
        control,
        model=None,
        eval_dataloader=None,
        processing_class=None,
        **kwargs,
    ):
        model.eval()
        dataset = eval_dataloader.dataset
        size_dataset = len(dataset)
        sample_indexes = range(0, size_dataset, size_dataset // (self.num_samples - 1))
        images_to_log = []
        mean = torch.as_tensor(processing_class.image_mean).unsqueeze(1).unsqueeze(1)
        std = torch.as_tensor(processing_class.image_std).unsqueeze(1).unsqueeze(1)
        target_classes = torch.tensor(list(self.id2label.keys()))

        for idx in sample_indexes:
            sample = dataset[idx]
            image = sample["pixel_values"]
            if self.denormalize:
                unprocessed_images = image * std + mean
                unprocessed_images = unprocessed_images.clip(0, 1).to(dtype=torch.float32)
            else:
                unprocessed_images = image
            inputs = {"pixel_values": image.unsqueeze(0)}
            targets = sample["labels"]

            inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            size = torch.as_tensor(unprocessed_images.shape[1:]).unsqueeze(0)
            results = processing_class.post_process_object_detection(
                outputs, target_sizes=size, threshold=self.threshold
            )[0]

            results = {k: v.cpu() for k, v in results.items()}

            mask = torch.isin(results["labels"], target_classes)
            results = {
                "boxes": results["boxes"][mask],
                "scores": results["scores"][mask],
                "labels": results["labels"][mask],
            }

            # Draw GT bounding boxes
            drawing = draw_bbox(
                unprocessed_images,
                targets,
                self.id2label,
                "yolo",
                boxes_key="boxes",
                labels_key="class_labels",
                scores_key=None,
                colors="red",
            )
            drawing = draw_bbox(
                drawing,
                results,
                self.id2label,
                "pascal_voc",
                boxes_key="boxes",
                labels_key="labels",
                scores_key="scores",
                colors="green",
            )

            images_to_log.append(
                wandb.Image(
                    drawing,
                    caption=f"Step {state.global_step} sample {idx}",
                )
            )

        wandb.log({"eval/visualizations": images_to_log}, step=state.global_step)
