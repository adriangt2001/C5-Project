from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from src.utils.kitti_dataset_motsio import KittiDataset

# mateixes funcions que al notebook 

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def random_point_from_mask(mask: np.ndarray, rng: np.random.Generator) -> list[int]:
    ys, xs = np.where(mask)
    idx = int(rng.integers(0, len(xs)))
    return [int(xs[idx]), int(ys[idx])]


def interior_point_from_mask(mask: np.ndarray) -> list[int]:
    mask_uint8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return [int(x), int(y)]


def random_edge_point_from_mask(mask: np.ndarray, rng: np.random.Generator) -> list[int]:
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel)
    edges = mask_uint8 - eroded

    ys, xs = np.where(edges)
    if len(xs) == 0:
        ys, xs = np.where(mask_uint8)

    idx = int(rng.integers(0, len(xs)))
    return [int(xs[idx]), int(ys[idx])]


@torch.no_grad()
def predict_mask(
    image_np: np.ndarray,
    point: list[int],
    processor: SamProcessor,
    model: SamModel,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(
        image_np,
        input_points=[[[point[0], point[1]]]],
        input_labels=[[1]],
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    return masks[0][0][0].numpy()


def save_overlay(
    image_np: np.ndarray,
    mask: np.ndarray,
    point: list[int] | None,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image_np)
    ax.imshow(mask, alpha=0.5)
    if point is not None:
        ax.scatter(point[0], point[1], c="red", s=40)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prompt_strategy_comparison(
    image_np: np.ndarray,
    gt_mask: np.ndarray,
    predictions: list[np.ndarray],
    points: list[list[int]],
    output_path: Path,
) -> None:
    titles = ["Edge Prompt", "Random Prompt", "Interior Prompt"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(3):
        axes[0, i].imshow(image_np)
        axes[0, i].imshow(gt_mask, alpha=0.5)
        axes[0, i].scatter(points[i][0], points[i][1], c="red", s=30)
        axes[0, i].set_title(f"{titles[i]} - GT")
        axes[0, i].axis("off")

        axes[1, i].imshow(image_np)
        axes[1, i].imshow(predictions[i], alpha=0.5)
        axes[1, i].scatter(points[i][0], points[i][1], c="red", s=30)
        axes[1, i].set_title(f"{titles[i]} - SAM")
        axes[1, i].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_dataset(args) -> KittiDataset:
    return KittiDataset(
        image_folder=args.image_folder,
        annotations_folder=args.annotations_folder,
        seqmap_file=args.seqmap_file,
        transforms=None,
    )


def get_prompt_comparison_sample(dataset: KittiDataset, start_index: int):
    for idx in range(start_index, len(dataset)):
        image, target = dataset[idx]
        if len(target["masks"]) > 0:
            return image, target

    for idx in range(0, start_index):
        image, target = dataset[idx]
        if len(target["masks"]) > 0:
            return image, target

    raise RuntimeError("No masks found in dataset for prompt comparison.")

# main task a --> mateix que el notebook pero guardant resultats i metrics 
def main_task_a(args) -> None:
    output_dir = Path(args.output_dir) / "task_a"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading SAM...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    dataset = build_dataset(args)
    print(f"Dataset size: {len(dataset)} samples")

    ious = []
    car_ious = []
    pedestrian_ious = []

    best_iou = -1.0
    worst_iou = 2.0
    best_sample = None
    worst_sample = None

    max_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))

    for idx in tqdm(range(max_samples), desc="Task A evaluation"):
        image, target = dataset[idx]
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        masks = target["masks"].numpy()
        labels = target["labels"].numpy()

        if len(masks) == 0:
            continue

        for gt_mask, label in zip(masks, labels):
            point = random_edge_point_from_mask(gt_mask, rng)
            pred_mask = predict_mask(image_np, point, processor, model, device)

            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            iou = compute_iou(pred_mask, gt_mask)
            ious.append(iou)

            if label == 1:
                car_ious.append(iou)
            elif label == 2:
                pedestrian_ious.append(iou)

            sample_data = (image_np.copy(), gt_mask.copy(), pred_mask.copy(), point)
            if iou > best_iou:
                best_iou = iou
                best_sample = sample_data
            if iou < worst_iou:
                worst_iou = iou
                worst_sample = sample_data

    if not ious:
        raise RuntimeError("Task A evaluation produced no objects. Check dataset paths and annotations.")

    metrics = {
        "num_images_evaluated": max_samples,
        "num_objects_evaluated": len(ious),
        "mean_iou": round(float(np.mean(ious)), 4),
        "mean_iou_cars": round(float(np.mean(car_ious)), 4) if car_ious else None,
        "mean_iou_pedestrians": round(float(np.mean(pedestrian_ious)), 4) if pedestrian_ious else None,
        "lowest_iou": round(float(np.min(ious)), 4),
        "highest_iou": round(float(np.max(ious)), 4),
    }

    metrics_path = output_dir / "task_a_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    best_img, best_gt, best_pred, best_point = best_sample
    worst_img, worst_gt, worst_pred, worst_point = worst_sample

    save_overlay(
        best_img,
        best_gt,
        best_point,
        "Task A Best Sample - Ground Truth",
        output_dir / "best_gt.png",
    )
    save_overlay(
        best_img,
        best_pred,
        best_point,
        f"Task A Best Sample - SAM Prediction (IoU={best_iou:.3f})",
        output_dir / "best_pred.png",
    )
    save_overlay(
        worst_img,
        worst_gt,
        worst_point,
        "Task A Worst Sample - Ground Truth",
        output_dir / "worst_gt.png",
    )
    save_overlay(
        worst_img,
        worst_pred,
        worst_point,
        f"Task A Worst Sample - SAM Prediction (IoU={worst_iou:.3f})",
        output_dir / "worst_pred.png",
    )

    prompt_start_index = min(args.prompt_comparison_index, len(dataset) - 1)
    prompt_image, prompt_target = get_prompt_comparison_sample(dataset, prompt_start_index)
    prompt_image_np = (prompt_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    prompt_mask = prompt_target["masks"][0].numpy()

    edge_point = random_edge_point_from_mask(prompt_mask, rng)
    random_point = random_point_from_mask(prompt_mask, rng)
    interior_point = interior_point_from_mask(prompt_mask)

    prompt_points = [edge_point, random_point, interior_point]
    prompt_predictions = [
        predict_mask(prompt_image_np, point, processor, model, device) for point in prompt_points
    ]

    save_prompt_strategy_comparison(
        prompt_image_np,
        prompt_mask,
        prompt_predictions,
        prompt_points,
        output_dir / "prompt_strategy_comparison.png",
    )

    print("\n=== Task A metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved outputs to: {output_dir}")
