from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor, pipeline

from src.utils.kitti_dataset_motsio import KittiDataset
from src.utils.metrics import InstanceSegmentationMetrics, parse_segmentation_metrics
from src.utils.visualizations import show_box, show_mask, show_points

POINT_VARIANTS = {
    "edge": "Point sampled on the object boundary",
    "interior": "Point sampled from the mask interior",
}

BBOX_VARIANTS = {
    "normal": "Ground-truth bbox",
    "slight_shift": "Slightly shifted bbox",
    "heavy_shift": "Strongly shifted bbox",
}

TEXT_VARIANTS = {
    "generic": {
        "car": ["car"],
        "pedestrian": ["person"],
        "description": "Short, generic prompts",
    },
    "specific": {
        "car": ["car on the road", "parked car"],
        "pedestrian": ["pedestrian", "person walking"],
        "description": "More specific prompts with context",
    },
    "broad": {
        "car": ["vehicle"],
        "pedestrian": ["human"],
        "description": "Broader prompts that may be less precise",
    },
}

FAMILY_TO_VARIANTS = {
    "point": POINT_VARIANTS,
    "bbox": BBOX_VARIANTS,
    "text": {key: value["description"] for key, value in TEXT_VARIANTS.items()},
}

ID2LABEL = {
    1: "car",
    2: "pedestrian",
}

# com les slides

FAMILY_COLORS = {
    "point": "#E7B051",
    "bbox": "#2A9D8F",
    "text": "#264653",
}


@dataclass
class VariantAccumulator:
    metric: InstanceSegmentationMetrics = field(
        default_factory=lambda: InstanceSegmentationMetrics(class_metrics=True)
    )
    total_time: float = 0.0
    detector_time: float = 0.0
    sam_time: float = 0.0
    total_images: int = 0
    sam_score_sum: float = 0.0
    sam_score_count: int = 0

    def update_scores(self, scores: torch.Tensor) -> None:
        if scores.numel() == 0:
            return
        self.sam_score_sum += float(scores.sum().item())
        self.sam_score_count += int(scores.numel())


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def empty_prediction(image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    # prediccio buida yata
    _, h, w = image_tensor.shape
    return {
        "masks": torch.zeros((0, h, w), dtype=torch.bool),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }

def masks_to_boxes(masks: torch.Tensor) -> np.ndarray:
    # convertir segmentation masks a boundig boxes
    boxes = []
    for mask in masks:
        ys, xs = torch.where(mask > 0)
        if ys.numel() == 0 or xs.numel() == 0:
            boxes.append([0.0, 0.0, 1.0, 1.0])
            continue
        x1 = float(xs.min().item())
        y1 = float(ys.min().item())
        x2 = float(xs.max().item() + 1)
        y2 = float(ys.max().item() + 1)
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32)


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    # check que bbox estigui a dins la imatge
    clipped = box.copy()
    clipped[0] = np.clip(clipped[0], 0, width - 1)
    clipped[1] = np.clip(clipped[1], 0, height - 1)
    clipped[2] = np.clip(clipped[2], clipped[0] + 1, width)
    clipped[3] = np.clip(clipped[3], clipped[1] + 1, height)
    return clipped.astype(np.float32)


def random_edge_point_from_mask(mask: np.ndarray, rng: random.Random) -> list[float]:
    # random point que estigui a la edge
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel)
    edges = mask_uint8 - eroded
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        ys, xs = np.where(mask_uint8 > 0)
    idx = rng.randrange(len(xs))
    return [float(xs[idx]), float(ys[idx])]


def interior_point_from_mask(mask: np.ndarray) -> list[float]:
    # agafar el point central
    mask_uint8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return [float(x), float(y)]


def build_point_variants(gt_masks: torch.Tensor, rng: random.Random):
    # point prompts 
    points = {}

    if len(gt_masks) == 0:
        for key in POINT_VARIANTS:
            points[key] = np.zeros((0, 2), dtype=np.float32)
        return points

    edge_points = []
    interior_points = []

    for mask in gt_masks.cpu().numpy():
        edge_points.append(random_edge_point_from_mask(mask, rng))
        interior_points.append(interior_point_from_mask(mask))

    points["edge"] = np.asarray(edge_points, dtype=np.float32)
    points["interior"] = np.asarray(interior_points, dtype=np.float32)
    return points


def build_bbox_variants(gt_boxes: np.ndarray, width: int, height: int, rng: random.Random):
    boxes = {}

    if len(gt_boxes) == 0:
        for key in BBOX_VARIANTS:
            boxes[key] = np.zeros((0, 4), dtype=np.float32)
        return boxes

    slight_shift_boxes = []
    heavy_shift_boxes = []

    for box in gt_boxes:
        x1, y1, x2, y2 = box
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        slight_shift_box = np.asarray(
            [
                cx - 0.52 * bw,
                cy - 0.52 * bh,
                cx + 0.58 * bw,
                cy + 0.58 * bh,
            ],
            dtype=np.float32,
        )
        slight_shift_boxes.append(clip_box(slight_shift_box, width, height))

        shift_x = rng.choice([-1.0, 1.0]) * 0.45 * bw
        shift_y = rng.choice([-1.0, 1.0]) * 0.45 * bh
        heavy_shift_box = np.asarray(
            [
                cx - 0.55 * bw + shift_x,
                cy - 0.55 * bh + shift_y,
                cx + 0.55 * bw + shift_x,
                cy + 0.55 * bh + shift_y,
            ],
            dtype=np.float32,
        )
        heavy_shift_boxes.append(clip_box(heavy_shift_box, width, height))

    boxes["normal"] = gt_boxes.astype(np.float32)
    boxes["slight_shift"] = np.asarray(slight_shift_boxes, dtype=np.float32)
    boxes["heavy_shift"] = np.asarray(heavy_shift_boxes, dtype=np.float32)
    return boxes

# --------------- run sam ------------------

@torch.no_grad()
def run_sam_on_boxes(image_tensor, boxes, processor, sam_model, device):
    h, w = image_tensor.shape[1:]
    if len(boxes) == 0:
        return (
            torch.zeros((0, h, w), dtype=torch.bool),
            torch.zeros((0,), dtype=torch.float32),
        )

    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    inputs = processor(
        images=image_np,
        input_boxes=[boxes.tolist()],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = sam_model(**inputs, multimask_output=False)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]

    masks = masks[:, 0] > 0
    sam_scores = outputs.iou_scores[0][:, 0].detach().cpu().float()
    return masks.bool(), sam_scores


@torch.no_grad()
def run_sam_on_points(image_tensor, points, processor, sam_model, device):
    h, w = image_tensor.shape[1:]
    if len(points) == 0:
        return (
            torch.zeros((0, h, w), dtype=torch.bool),
            torch.zeros((0,), dtype=torch.float32),
        )

    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    prompt_points = points[:, None, :].astype(np.float32)
    prompt_labels = np.ones((len(points), 1), dtype=np.int64)

    inputs = processor(
        images=image_np,
        input_points=[prompt_points.tolist()],
        input_labels=[prompt_labels.tolist()],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = sam_model(**inputs, multimask_output=False)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]

    masks = masks[:, 0] > 0
    sam_scores = outputs.iou_scores[0][:, 0].detach().cpu().float()
    return masks.bool(), sam_scores


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.clip(union, a_min=1e-6, a_max=None)


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    # per cada box eliminar les boxes que tinguin un IoU > threshold
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        current = order[0]
        keep.append(current)
        if len(order) == 1:
            break
        ious = compute_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious < threshold]

    return np.asarray(keep, dtype=np.int64)


@torch.no_grad()
def run_text_variant(
    image_tensor,
    variant_key,
    detector,
    box_threshold,
    nms_threshold,
):
    prompts = TEXT_VARIANTS[variant_key]
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)

    detector_time = 0.0
    boxes = []
    labels = []
    scores = []

    for class_id, class_name in ((1, "car"), (2, "pedestrian")):
        candidate_prompts = prompts[class_name]
        class_boxes = []
        class_scores = []

        for prompt_text in candidate_prompts:
            start = time.perf_counter()
            results = detector(
                image_pil,
                candidate_labels=[prompt_text if prompt_text.endswith(".") else f"{prompt_text}."],
                threshold=box_threshold,
            )
            detector_time += time.perf_counter() - start

            for result in results:
                box = result["box"]
                class_boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
                class_scores.append(result["score"])

        if len(class_boxes) == 0:
            continue

        class_boxes = np.asarray(class_boxes, dtype=np.float32)
        class_scores = np.asarray(class_scores, dtype=np.float32)
        keep = greedy_nms(class_boxes, class_scores, nms_threshold)

        for idx in keep:
            boxes.append(class_boxes[idx])
            labels.append(class_id)
            scores.append(class_scores[idx])

    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            detector_time,
        )

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
        detector_time,
    )


def render_family_panel(
    family: str,
    image_tensor: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_boxes: np.ndarray,
    gt_labels: torch.Tensor,
    family_predictions: dict,
    save_path: Path,
):
    # nomes per visualitzar 
    variants = list(FAMILY_TO_VARIANTS[family].keys())
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, len(variants) + 1, figsize=(5 * (len(variants) + 1), 5))

    axes[0].imshow(image_np)
    for mask in gt_masks:
        show_mask(mask.cpu().numpy(), axes[0], random_color=True)
    for box in gt_boxes:
        show_box(box, axes[0])
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    for axis, variant_key in zip(axes[1:], variants):
        variant_data = family_predictions[variant_key]
        axis.imshow(image_np)

        for mask in variant_data["masks"]:
            show_mask(mask.cpu().numpy(), axis, random_color=True)

        if family == "point":
            prompt_points = variant_data["prompts"]
            if len(prompt_points) > 0:
                show_points(
                    np.asarray(prompt_points, dtype=np.float32),
                    np.ones(len(prompt_points), dtype=np.int64),
                    axis,
                    marker_size=120,
                )
        else:
            prompt_boxes = variant_data["prompts"]
            for box in prompt_boxes:
                show_box(box, axis)

        sam_score = variant_data["scores"].mean().item() if variant_data["scores"].numel() > 0 else 0.0
        axis.set_title(f"{variant_key}\nSAM score={sam_score:.3f}")
        axis.axis("off")

    fig.suptitle(f"{family.upper()} prompt analysis")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# mes visualitzacio 
def plot_grouped_metric(results: dict, metric_key: str, title: str, ylabel: str, save_path: Path):
    families = list(results["families"].keys())
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(families))
    width = 0.22

    for family_index, family in enumerate(families):
        variants = list(results["families"][family].keys())
        offsets = np.linspace(-width, width, num=len(variants))

        for offset, variant in zip(offsets, variants):
            value = results["families"][family][variant][metric_key]
            ax.bar(
                x[family_index] + offset,
                value,
                width=width * 0.9,
                color=FAMILY_COLORS[family],
                alpha=0.50 + 0.15 * (list(variants).index(variant)),
                edgecolor="black",
                linewidth=0.8,
            )
            ax.text(
                x[family_index] + offset,
                value + 0.01,
                variant,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([family.upper() for family in families], fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_speed_accuracy_scatter(results: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for family, family_results in results["families"].items():
        for variant, metrics in family_results.items():
            x = metrics["avg_time_per_image"]
            y = metrics["map"]
            ax.scatter(
                x,
                y,
                s=130,
                color=FAMILY_COLORS[family],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.text(x + 0.003, y + 0.008, f"{family}:{variant}", fontsize=8)

    ax.set_xlabel("Time per image (s)")
    ax.set_ylabel("mAP")
    ax.set_title("Accuracy vs Efficiency", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_classwise_map(results: dict, save_path: Path):
    families = list(results["families"].keys())
    fig, axes = plt.subplots(1, len(families), figsize=(12, 4.8), sharey=True)

    for ax, family in zip(axes, families):
        variants = list(results["families"][family].keys())
        car_values = [results["families"][family][variant]["map_car"] for variant in variants]
        ped_values = [results["families"][family][variant]["map_pedestrian"] for variant in variants]
        xpos = np.arange(len(variants))

        ax.bar(xpos - 0.18, car_values, width=0.36, color="#457B9D", label="Car")
        ax.bar(xpos + 0.18, ped_values, width=0.36, color="#A8DADC", label="Pedestrian")
        ax.set_xticks(xpos)
        ax.set_xticklabels(variants, rotation=20)
        ax.set_title(family.upper(), fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("mAP")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Class-wise mAP by Prompt Variant", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_result_visualizations(results: dict, output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_grouped_metric(
        results=results,
        metric_key="map",
        title="Prompt Family Comparison by mAP",
        ylabel="mAP",
        save_path=plots_dir / "map_by_variant.png",
    )
    plot_grouped_metric(
        results=results,
        metric_key="avg_time_per_image",
        title="Inference Time by Prompt Variant",
        ylabel="Seconds per image",
        save_path=plots_dir / "time_by_variant.png",
    )
    plot_classwise_map(
        results=results,
        save_path=plots_dir / "classwise_map.png",
    )
    plot_speed_accuracy_scatter(
        results=results,
        save_path=plots_dir / "accuracy_vs_efficiency.png",
    )
    return plots_dir


def compute_family_predictions(
    image_tensor: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes: np.ndarray,
    rng: random.Random,
    processor,
    sam_model,
    detector,
    device,
    box_threshold: float,
    nms_threshold: float,
):
    _, height, width = image_tensor.shape
    point_prompts = build_point_variants(gt_masks, rng)
    bbox_prompts = build_bbox_variants(gt_boxes, width, height, rng)
    family_predictions = {}

    for variant_key, prompt_points in point_prompts.items():
        pred_masks, sam_scores = run_sam_on_points(
            image_tensor=image_tensor,
            points=prompt_points,
            processor=processor,
            sam_model=sam_model,
            device=device,
        )
        pred = empty_prediction(image_tensor)
        if len(prompt_points) > 0:
            pred = {
                "masks": pred_masks,
                "scores": sam_scores,
                "labels": gt_labels.clone(),
            }
        family_predictions.setdefault("point", {})[variant_key] = {
            "masks": pred["masks"],
            "scores": pred["scores"],
            "prompts": prompt_points,
        }

    for variant_key, prompt_boxes in bbox_prompts.items():
        pred_masks, sam_scores = run_sam_on_boxes(
            image_tensor=image_tensor,
            boxes=prompt_boxes,
            processor=processor,
            sam_model=sam_model,
            device=device,
        )
        pred = empty_prediction(image_tensor)
        if len(prompt_boxes) > 0:
            pred = {
                "masks": pred_masks,
                "scores": sam_scores,
                "labels": gt_labels.clone(),
            }
        family_predictions.setdefault("bbox", {})[variant_key] = {
            "masks": pred["masks"],
            "scores": pred["scores"],
            "prompts": prompt_boxes,
        }

    for variant_key in TEXT_VARIANTS:
        boxes, labels_np, det_scores_np, _ = run_text_variant(
            image_tensor=image_tensor,
            variant_key=variant_key,
            detector=detector,
            box_threshold=box_threshold,
            nms_threshold=nms_threshold,
        )
        pred_masks, sam_scores = run_sam_on_boxes(
            image_tensor=image_tensor,
            boxes=boxes,
            processor=processor,
            sam_model=sam_model,
            device=device,
        )
        pred = empty_prediction(image_tensor)
        if len(boxes) > 0:
            det_scores = torch.tensor(det_scores_np, dtype=torch.float32)
            pred = {
                "masks": pred_masks,
                "scores": det_scores * sam_scores,
                "labels": torch.tensor(labels_np, dtype=torch.int64),
            }
        family_predictions.setdefault("text", {})[variant_key] = {
            "masks": pred["masks"],
            "scores": pred["scores"],
            "prompts": boxes,
        }

    return family_predictions


def generate_qualitative_visualizations(
    dataset,
    qualitative_dir: Path,
    rng: random.Random,
    max_visualizations: int,
    processor,
    sam_model,
    detector,
    device,
    box_threshold: float,
    nms_threshold: float,
):
    image_ids = list(range(len(dataset)))
    selected_ids = sorted(rng.sample(image_ids, k=min(max_visualizations, len(image_ids))))
  
    for image_id in tqdm(selected_ids, desc="Task G qualitative"):
        image_tensor, target = dataset[image_id]
        gt_masks = target["masks"].bool()
        if len(gt_masks) == 0:
            continue

        gt_labels = target["labels"].long()
        gt_boxes = masks_to_boxes(gt_masks)
        family_predictions = compute_family_predictions(
            image_tensor=image_tensor,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            gt_boxes=gt_boxes,
            rng=rng,
            processor=processor,
            sam_model=sam_model,
            detector=detector,
            device=device,
            box_threshold=box_threshold,
            nms_threshold=nms_threshold,
        )

        for family in FAMILY_TO_VARIANTS:
            render_family_panel(
                family=family,
                image_tensor=image_tensor.cpu(),
                gt_masks=gt_masks.cpu(),
                gt_boxes=gt_boxes,
                gt_labels=gt_labels.cpu(),
                family_predictions=family_predictions[family],
                save_path=qualitative_dir / f"{family}_image_{image_id:04d}.png",
            )


def finalize_results(accumulator: VariantAccumulator) -> dict:
    metrics = parse_segmentation_metrics(accumulator.metric.compute(), ID2LABEL)
    avg_time = accumulator.total_time / max(accumulator.total_images, 1)
    avg_detector = accumulator.detector_time / max(accumulator.total_images, 1)
    avg_sam = accumulator.sam_time / max(accumulator.total_images, 1)
    avg_sam_score = accumulator.sam_score_sum / max(accumulator.sam_score_count, 1)

    metrics.update(
        {
            "avg_time_per_image": round(avg_time, 6),
            "avg_detector_time_per_image": round(avg_detector, 6),
            "avg_sam_time_per_image": round(avg_sam, 6),
            "fps": round(1.0 / avg_time, 4) if avg_time > 0 else 0.0,
            "avg_sam_score": round(avg_sam_score, 4),
            "images_evaluated": accumulator.total_images,
        }
    )
    return metrics

# main per la task g agafa args i fa tot el processament, visualitzacions, etc
def main_task_g(args):
    image_folder = args.image_folder
    annotations_folder = args.annotations_folder
    seqmap_file = args.seqmap_file
    batch_size = args.batch_size
    box_threshold = args.box_threshold
    nms_threshold = args.nms_threshold
    max_visualizations = args.max_visualizations
    output_dir = Path(args.output_dir)
    seed = args.seed
    visualize_only = getattr(args, "visualize_only", False)
    generate_qualitative = getattr(args, "generate_qualitative", True)

    output_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = Path(
        getattr(args, "metrics_json", output_dir / "prompt_analysis_metrics.json")
    )

    if visualize_only:
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        plots_dir = generate_result_visualizations(results, output_dir)
        print(f"Generated plots from {metrics_json_path}")

        if generate_qualitative:
            rng = random.Random(seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            detector_device = 0 if device.type == "cuda" else -1

            dataset = KittiDataset(
                image_folder=image_folder,
                annotations_folder=annotations_folder,
                seqmap_file=seqmap_file,
                transforms=None,
            )

            print("Loading GroundingDINO for qualitative panels...")
            detector = pipeline(
                model="IDEA-Research/grounding-dino-tiny",
                task="zero-shot-object-detection",
                device=detector_device,
            )

            print("Loading SAM for qualitative panels...")
            processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

            generate_qualitative_visualizations(
                dataset=dataset,
                qualitative_dir=qualitative_dir,
                rng=rng,
                max_visualizations=max_visualizations,
                processor=processor,
                sam_model=sam_model,
                detector=detector,
                device=device,
                box_threshold=box_threshold,
                nms_threshold=nms_threshold,
            )

        print(f"Saved presentation plots to {plots_dir}")
        if generate_qualitative:
            print(f"Saved qualitative panels to {qualitative_dir}")
        return

    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_device = 0 if device.type == "cuda" else -1

    dataset = KittiDataset(
        image_folder=image_folder,
        annotations_folder=annotations_folder,
        seqmap_file=seqmap_file,
        transforms=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    image_ids = list(range(len(dataset)))
    selected_ids = set(rng.sample(image_ids, k=min(max_visualizations, len(image_ids))))

    print("Loading GroundingDINO...")
    detector = pipeline(
        model="IDEA-Research/grounding-dino-tiny",
        task="zero-shot-object-detection",
        device=detector_device,
    )

    print("Loading SAM...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    accumulators = {
        family: {variant: VariantAccumulator() for variant in variants}
        for family, variants in FAMILY_TO_VARIANTS.items()
    }
    qualitative_cache = {}

    for images, targets in tqdm(loader, desc="Task G prompt analysis"):
        for image_tensor, target in zip(images, targets):
            image_id = int(target["image_id"])
            _, height, width = image_tensor.shape
            gt_masks = target["masks"].bool()
            gt_labels = target["labels"].long()
            gt = {
                "masks": gt_masks,
                "labels": gt_labels,
            }

            gt_boxes = masks_to_boxes(gt_masks)
            point_prompts = build_point_variants(gt_masks, rng)
            bbox_prompts = build_bbox_variants(gt_boxes, width, height, rng)

            family_predictions = {}

            for variant_key, prompt_points in point_prompts.items():
                acc = accumulators["point"][variant_key]
                acc.total_images += 1

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                pred_masks, sam_scores = run_sam_on_points(
                    image_tensor=image_tensor,
                    points=prompt_points,
                    processor=processor,
                    sam_model=sam_model,
                    device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

                acc.total_time += end - start
                acc.sam_time += end - start
                acc.update_scores(sam_scores)

                pred = empty_prediction(image_tensor)
                if len(prompt_points) > 0:
                    pred = {
                        "masks": pred_masks,
                        "scores": sam_scores,
                        "labels": gt_labels.clone(),
                    }

                acc.metric.update([pred], [gt])
                family_predictions.setdefault("point", {})[variant_key] = {
                    "masks": pred["masks"],
                    "scores": pred["scores"],
                    "prompts": prompt_points,
                }

            for variant_key, prompt_boxes in bbox_prompts.items():
                acc = accumulators["bbox"][variant_key]
                acc.total_images += 1

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                pred_masks, sam_scores = run_sam_on_boxes(
                    image_tensor=image_tensor,
                    boxes=prompt_boxes,
                    processor=processor,
                    sam_model=sam_model,
                    device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

                acc.total_time += end - start
                acc.sam_time += end - start
                acc.update_scores(sam_scores)

                pred = empty_prediction(image_tensor)
                if len(prompt_boxes) > 0:
                    pred = {
                        "masks": pred_masks,
                        "scores": sam_scores,
                        "labels": gt_labels.clone(),
                    }

                acc.metric.update([pred], [gt])
                family_predictions.setdefault("bbox", {})[variant_key] = {
                    "masks": pred["masks"],
                    "scores": pred["scores"],
                    "prompts": prompt_boxes,
                }

            for variant_key in TEXT_VARIANTS:
                acc = accumulators["text"][variant_key]
                acc.total_images += 1

                start = time.perf_counter()
                boxes, labels_np, det_scores_np, detector_time = run_text_variant(
                    image_tensor=image_tensor,
                    variant_key=variant_key,
                    detector=detector,
                    box_threshold=box_threshold,
                    nms_threshold=nms_threshold,
                )
                acc.detector_time += detector_time

                if device.type == "cuda":
                    torch.cuda.synchronize()
                sam_start = time.perf_counter()
                pred_masks, sam_scores = run_sam_on_boxes(
                    image_tensor=image_tensor,
                    boxes=boxes,
                    processor=processor,
                    sam_model=sam_model,
                    device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                sam_end = time.perf_counter()

                total_end = time.perf_counter()
                acc.total_time += total_end - start
                acc.sam_time += sam_end - sam_start
                acc.update_scores(sam_scores)

                pred = empty_prediction(image_tensor)
                if len(boxes) > 0:
                    det_scores = torch.tensor(det_scores_np, dtype=torch.float32)
                    pred = {
                        "masks": pred_masks,
                        "scores": det_scores * sam_scores,
                        "labels": torch.tensor(labels_np, dtype=torch.int64),
                    }

                acc.metric.update([pred], [gt])
                family_predictions.setdefault("text", {})[variant_key] = {
                    "masks": pred["masks"],
                    "scores": pred["scores"],
                    "prompts": boxes,
                }

            if image_id in selected_ids and len(gt_masks) > 0:
                qualitative_cache[image_id] = {
                    "image_tensor": image_tensor.cpu(),
                    "gt_masks": gt_masks.cpu(),
                    "gt_boxes": gt_boxes,
                    "gt_labels": gt_labels.cpu(),
                    "predictions": family_predictions,
                }

    results = {"families": {}}
    for family, family_accumulators in accumulators.items():
        results["families"][family] = {}
        for variant, accumulator in family_accumulators.items():
            results["families"][family][variant] = finalize_results(accumulator)

    for image_id, sample in qualitative_cache.items():
        for family in FAMILY_TO_VARIANTS:
            render_family_panel(
                family=family,
                image_tensor=sample["image_tensor"],
                gt_masks=sample["gt_masks"],
                gt_boxes=sample["gt_boxes"],
                gt_labels=sample["gt_labels"],
                family_predictions=sample["predictions"][family],
                save_path=qualitative_dir / f"{family}_image_{image_id:04d}.png",
            )

    with open(output_dir / "prompt_analysis_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plots_dir = generate_result_visualizations(results, output_dir)

    print("\n=== Task G prompt analysis ===")
    for family, family_results in results["families"].items():
        print(f"\n[{family}]")
        for variant, metrics in family_results.items():
            print(
                f"- {variant}: map={metrics['map']:.4f}, "
                f"avg_sam_score={metrics['avg_sam_score']:.4f}, "
                f"time/img={metrics['avg_time_per_image']:.4f}s"
            )

    print(f"\nSaved metrics to {output_dir / 'prompt_analysis_metrics.json'}")
    print(f"Saved qualitative panels to {qualitative_dir}")
    print(f"Saved presentation plots to {plots_dir}")
