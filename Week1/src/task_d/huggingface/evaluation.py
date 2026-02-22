import argparse
from functools import partial

import numpy as np
import torch
import wandb
from huggingface_hub import interpreter_login
from src.custom_datasets import KittiDatasetHuggingface
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (AutoImageProcessor, AutoModelForObjectDetection,
                          Trainer, TrainingArguments)
from transformers.image_transforms import center_to_corners_format


class ModelOutput:
    @torch.no_grad()
    def __init__(self, logits, pred_boxes):
        self.logits: torch.Tensor = logits
        self.pred_boxes: torch.Tensor = pred_boxes

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='facebook/detr-resnet-50')
    parser.add_argument('--annotation_folder', type=str, default='instances_txt')
    parser.add_argument('--image_folder', type=str, default='training')

    args = parser.parse_args()
    return args

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, id2label, threshold=0.0):
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
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes,
        )
        # Iterate through the predictions for each image in the batch
        for pred_dict in post_processed_output:
            # Create a mask where True means the label is either 1 (person) or 3 (car)
            mask = torch.isin(pred_dict["labels"], target_classes)
            
            # Apply the mask to boxes, scores, and labels to keep only classes 1 and 3
            filtered_pred = {
                "boxes": pred_dict["boxes"][mask],
                "scores": pred_dict["scores"][mask],
                "labels": pred_dict["labels"][mask]
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
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
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

def augment_and_transform_batch(examples, image_processor, class_map, return_pixel_mask=False):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))
        images.append(image)

        # map from dataset to model labels
        objects["category"] = [class_map[x] for x in objects["category"]]

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, objects["category"], objects["area"], objects["bbox"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result

def evaluation(args):
    interpreter_login()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.variant
    dataset = args.dataset
    annotation_folder = args.annotation_folder
    image_folder = args.image_folder
    batch_size = args.batch_size
    log_wandb = args.log_wandb

    # Load Model
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.to(device)

    # Load Dataset
    dataset = KittiDatasetHuggingface(dataset, annotation_folder, image_folder, 'src/custom_datasets/val.seqmap').get_hf_ds()
    
    data2model = {
        1: model.config.label2id['car'],
        2: model.config.label2id['person']
    }

    id2label = {
        model.config.label2id['car']: 'car',
        model.config.label2id['person']: 'pedestrian'
    }
    label2id = {v: k for k, v in id2label.items()}

    image_processor = AutoImageProcessor.from_pretrained(
        model_name
    )

    validation_transform_batch = partial(
        augment_and_transform_batch, image_processor=image_processor, class_map=data2model
    )

    dataset = dataset.with_transform(validation_transform_batch)


    # Define Metrics
    eval_compute_metrics_fn = partial(
        compute_metrics, image_processor=image_processor, threshold=0.5, id2label=id2label
    )

    # Initialize wandb
    if log_wandb:
        wandb.init(
            project="C5-Week1", 
            entity="c5-team2", 
            name=f"Eval-{args.model}",
            config=args
        )

    # Trainer
    training_args = TrainingArguments(
        output_dir="results/task_d",
        num_train_epochs=30,
        fp16=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        run_name="pretrained-detr-eval",
        eval_do_concat_batches=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix='eval')

    # Create a new dictionary for custom W&B names
    metrics_to_log = {
        "mAP/main": metrics["eval_map"],          # AP @.50:.95
        "mAP/50": metrics["eval_map_50"],         # AP @.50
        "mAP/75": metrics["eval_map_75"],         # AP @.75
        "mAP/class_Car": metrics["eval_map_class_Car"],
        "mAP/class_Pedestrian": metrics["eval_map_class_Pedestrian"],
        "mAP/small": metrics["eval_map_small"],
        "mAP/medium": metrics["eval_map_medium"],
        "mAP/large": metrics["eval_map_large"],
        "mAR/Det1": metrics["eval_mar_1"],        # AR
        "mAR/Det10": metrics["eval_mar_10"],
        "mAR/Det100": metrics["eval_mar_100"],
        "mAR/Det100_class_Car": metrics["eval_mar_100_class_Car"],
        "mAR/Det100_class_Pedestrian": metrics["eval_mar_100_class_Pedestrian"],
        "mAR/small": metrics["eval_mar_small"],
        "mAR/medium": metrics["eval_mar_medium"],
        "mAR/large": metrics["eval_mar_large"],
        "performance/total_time": metrics["eval_runtime"],
        "performance/fps": metrics["eval_samples_per_second"],
        "performance/avg_time_per_img": 1 / metrics["eval_samples_per_second"]
    }

    # Log your perfectly formatted metrics directly
    if log_wandb:
        wandb.log(metrics_to_log)
        wandb.finish()

    print(f"\nEvaluation Finished for {args.model}:")
    print(f"mAP @.50:.95:  {metrics['eval_map']:.4f}")
    print(f"mAP @.50:      {metrics['eval_map_50']:.4f}")
    print(f"mAP @.75:      {metrics['eval_map_75']:.4f}")
    print(f"mAR Det1:      {metrics['eval_mar_1']:.4f}")
    print(f"mAR Det10:     {metrics['eval_mar_10']:.4f}")
    print(f"mAR Det100:    {metrics['eval_mar_100']:.4f}")

if __name__ == '__main__':
    args = args_parser()

    evaluation(args)