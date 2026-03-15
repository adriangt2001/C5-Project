from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

from src.utils.kitti_dataset_motsio2 import KittiDataset

PROMPT_SETS = [
    {
        "name": "baseline",
        "car_prompts": ["car"],
        "pedestrian_prompts": ["person"],
    },
    {
        "name": "synonyms",
        "car_prompts": ["car", "automobile", "vehicle"],
        "pedestrian_prompts": ["person", "pedestrian", "human"],
    },
    {
        "name": "context",
        "car_prompts": ["car on road", "road vehicle"],
        "pedestrian_prompts": ["person walking", "walking pedestrian"],
    },
]

# mateixes funcions que al notebook 
def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_a = max(float(box_a[0]), float(box_b[0]))
    y_a = max(float(box_a[1]), float(box_b[1]))
    x_b = min(float(box_a[2]), float(box_b[2]))
    y_b = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter

    if union == 0:
        return 0.0

    return float(inter / union)


def build_dataset(args) -> KittiDataset:
    return KittiDataset(
        image_folder=args.image_folder,
        annotations_folder=args.annotations_folder,
        seqmap_file=args.seqmap_file,
        transforms=None,
        only_mask=True,
    )


def load_dino(device: torch.device):
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)
    model.eval()
    return processor, model


def load_sam(device: torch.device):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    return processor, model


@torch.no_grad()
def run_dino(
    image_np: np.ndarray,
    prompts: list[str],
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
):
    text = ". ".join(prompts)
    image_pil = Image.fromarray((image_np * 255).astype("uint8"))

    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )[0]

    boxes = results["boxes"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()
    labels = [label.replace(".", "").strip().lower() for label in results["labels"]]
    return boxes, scores, labels


def evaluate_prompt_set(
    dataset: KittiDataset,
    prompt_set: dict,
    dino_processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
    max_samples: int,
):
    tp = 0
    fp = 0
    fn = 0

    all_scores = []
    all_matches = []
    total_gt = 0

    prompts = prompt_set["car_prompts"] + prompt_set["pedestrian_prompts"]

    for idx in tqdm(range(max_samples), desc=f"Task B {prompt_set['name']}"):
        image, target = dataset[idx]
        image_np = image.permute(1, 2, 0).numpy()
        gt_boxes = target["boxes"].numpy()

        total_gt += len(gt_boxes)
        pred_boxes, pred_scores, _ = run_dino(
            image_np,
            prompts,
            dino_processor,
            dino_model,
            device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        matched_gt = set()

        for pred_box in pred_boxes:
            best_iou = 0.0
            best_gt = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            if best_iou >= 0.5 and best_gt not in matched_gt:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1

        matched_gt_for_ap = set()
        for pred_box, score in zip(pred_boxes, pred_scores):
            best_iou = 0.0
            best_gt = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            if best_iou >= 0.5 and best_gt not in matched_gt_for_ap:
                match = 1
                matched_gt_for_ap.add(best_gt)
            else:
                match = 0

            all_scores.append(float(score))
            all_matches.append(match)

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    if all_scores:
        order = np.argsort(-np.array(all_scores))
        matches = np.array(all_matches)[order]
        tp_curve = np.cumsum(matches)
        fp_curve = np.cumsum(1 - matches)
        precision_curve = tp_curve / (tp_curve + fp_curve + 1e-6)
        recall_curve = tp_curve / (total_gt + 1e-6)
        ap50 = float(np.trapz(precision_curve, recall_curve))
    else:
        ap50 = 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "ap50": round(float(ap50), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "num_gt_boxes": int(total_gt),
    }


def find_sample_with_boxes(dataset: KittiDataset, start_index: int):
    for idx in range(start_index, len(dataset)):
        image, target = dataset[idx]
        if len(target["boxes"]) > 0:
            return idx, image, target

    for idx in range(0, start_index):
        image, target = dataset[idx]
        if len(target["boxes"]) > 0:
            return idx, image, target

    raise RuntimeError("No boxes found in dataset for Task B visualization.")


def save_prompt_comparison(
    image_np: np.ndarray,
    prompt_boxes: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(prompt_boxes), 1, figsize=(18, 6))
    if len(prompt_boxes) == 1:
        axes = [axes]

    for ax, (prompt_name, boxes) in zip(axes, prompt_boxes.items()):
        ax.imshow(image_np)
        ax.set_title(prompt_name)

        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_sam_for_boxes(
    image_pil: Image.Image,
    boxes: np.ndarray,
    processor: SamProcessor,
    model: SamModel,
    device: torch.device,
) -> list[np.ndarray]:
    masks = []
    for box in boxes:
        sam_inputs = processor(
            image_pil,
            input_boxes=[[box.tolist()]],
            return_tensors="pt",
        ).to(device)

        sam_outputs = model(**sam_inputs)
        mask = processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0][0][0].numpy()
        masks.append(mask)

    return masks


def save_sam_segmentation(
    image_np: np.ndarray,
    masks: list[np.ndarray],
    output_path: Path,
) -> None:
    result_img = image_np.copy()
    for mask in masks:
        result_img[mask.astype(bool)] = [255, 255, 0]

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(result_img)
    ax.set_title("SAM Segmentation from DINO Boxes")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main_task_b(args) -> None:
    output_dir = Path(args.output_dir) / "task_b"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    max_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))

    print("Loading GroundingDINO...")
    dino_processor, dino_model = load_dino(device)

    metrics = {}
    for prompt_set in PROMPT_SETS:
        metrics[prompt_set["name"]] = evaluate_prompt_set(
            dataset=dataset,
            prompt_set=prompt_set,
            dino_processor=dino_processor,
            dino_model=dino_model,
            device=device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_samples=max_samples,
        )

    metrics["config"] = {
        "max_samples": max_samples,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }

    metrics_path = output_dir / "task_b_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    prompt_index = min(args.prompt_comparison_index, len(dataset) - 1)
    sample_idx, sample_image, _ = find_sample_with_boxes(dataset, prompt_index)
    sample_image_np = sample_image.permute(1, 2, 0).numpy()

    prompt_boxes = {}
    for prompt_set in PROMPT_SETS:
        prompts = prompt_set["car_prompts"] + prompt_set["pedestrian_prompts"]
        boxes, _, _ = run_dino(
            sample_image_np,
            prompts,
            dino_processor,
            dino_model,
            device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        prompt_boxes[prompt_set["name"]] = boxes

    save_prompt_comparison(
        sample_image_np,
        prompt_boxes,
        output_dir / f"prompt_comparison_sample_{sample_idx}.png",
    )

    print("Loading SAM for Task B qualitative output...")
    sam_processor, sam_model = load_sam(device)

    baseline_boxes = prompt_boxes["baseline"]
    image_pil = Image.fromarray((sample_image_np * 255).astype("uint8"))
    sam_masks = run_sam_for_boxes(
        image_pil,
        baseline_boxes,
        sam_processor,
        sam_model,
        device,
    )
    save_sam_segmentation(
        sample_image_np,
        sam_masks,
        output_dir / f"sam_from_dino_sample_{sample_idx}.png",
    )

    print("\n=== Task B metrics ===")
    for prompt_name in ["baseline", "synonyms", "context"]:
        print(f"{prompt_name}: {metrics[prompt_name]}")
    print(f"Saved outputs to: {output_dir}")
