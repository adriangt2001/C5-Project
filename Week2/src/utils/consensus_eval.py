from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class Detection:
    prompt_idx: int
    class_name: str
    score: float
    box: np.ndarray
    mask: Optional[np.ndarray] = None


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return np.logical_and(a, b).sum() / union


def detection_iou(det_a: Detection, det_b: Detection, use_masks: bool) -> float:
    if use_masks and det_a.mask is not None and det_b.mask is not None:
        return mask_iou(det_a.mask, det_b.mask)
    return box_iou_xyxy(det_a.box, det_b.box)


def _post_process_masks(sam_processor, sam_inputs, sam_outputs) -> np.ndarray:
    return sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0][0][0].numpy()


def run_grounded_sam_for_class(
    image_np: np.ndarray,
    prompts: Sequence[str],
    class_name: str,
    dino_processor,
    dino_model,
    sam_processor,
    sam_model,
    device: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    max_detections: Optional[int] = None,
    return_masks: bool = True,
) -> List[Dict[str, Any]]:
    text_labels = [list(prompts)]
    inputs = dino_processor(images=image_np, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    detections = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_np.shape[:2]],
    )[0]

    boxes = detections["boxes"].detach().cpu().numpy() if len(detections["boxes"]) else np.zeros((0, 4))
    scores = detections["scores"].detach().cpu().numpy() if len(detections["scores"]) else np.zeros((0,))

    ranked_indices = np.argsort(-scores)
    if max_detections is not None:
        ranked_indices = ranked_indices[:max_detections]

    results: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        box = boxes[idx].astype(np.float32)
        score = float(scores[idx])
        record: Dict[str, Any] = {
            "class_name": class_name,
            "score": score,
            "box": box,
        }

        if return_masks:
            input_boxes = [[[box.tolist()]]]
            sam_inputs = sam_processor(image_np, input_boxes=input_boxes, return_tensors="pt").to(device)
            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs)
            record["mask"] = _post_process_masks(sam_processor, sam_inputs, sam_outputs)

        results.append(record)

    return results


def run_prompt_set_on_image(
    image,
    prompt_idx: int,
    prompt_set: Dict[str, Sequence[str]],
    dino_processor,
    dino_model,
    sam_processor,
    sam_model,
    device: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    max_detections_per_class: Optional[int] = None,
    return_masks: bool = True,
) -> List[Detection]:
    if isinstance(image, torch.Tensor):
        image_np = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        image_np = np.asarray(image)
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)

    detections: List[Detection] = []
    class_to_prompts = {
        "car": prompt_set["car_prompts"],
        "pedestrian": prompt_set["pedestrian_prompts"],
    }

    for class_name, prompts in class_to_prompts.items():
        class_detections = run_grounded_sam_for_class(
            image_np=image_np,
            prompts=prompts,
            class_name=class_name,
            dino_processor=dino_processor,
            dino_model=dino_model,
            sam_processor=sam_processor,
            sam_model=sam_model,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_detections=max_detections_per_class,
            return_masks=return_masks,
        )
        for det in class_detections:
            detections.append(
                Detection(
                    prompt_idx=prompt_idx,
                    class_name=class_name,
                    score=det["score"],
                    box=np.asarray(det["box"], dtype=np.float32),
                    mask=det.get("mask"),
                )
            )

    return detections


def build_consensus_clusters(
    detections: Iterable[Detection],
    min_votes: int = 2,
    iou_threshold: float = 0.5,
    use_masks: bool = True,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Detection]] = {"car": [], "pedestrian": []}
    for det in detections:
        grouped.setdefault(det.class_name, []).append(det)

    consensus_clusters: List[Dict[str, Any]] = []

    for class_name, class_detections in grouped.items():
        ordered = sorted(class_detections, key=lambda det: det.score, reverse=True)
        clusters: List[Dict[str, Any]] = []

        for det in ordered:
            best_cluster = None
            best_iou = -1.0
            for cluster in clusters:
                iou = detection_iou(det, cluster["representative"], use_masks=use_masks)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_cluster = cluster

            if best_cluster is None:
                clusters.append(
                    {
                        "class_name": class_name,
                        "members": [det],
                        "prompt_votes": {det.prompt_idx},
                        "representative": det,
                    }
                )
                continue

            best_cluster["members"].append(det)
            best_cluster["prompt_votes"].add(det.prompt_idx)

            members = best_cluster["members"]
            mean_box = np.mean(np.stack([member.box for member in members], axis=0), axis=0)
            mean_score = float(np.mean([member.score for member in members]))
            representative_mask = best_cluster["representative"].mask
            if use_masks:
                mask_candidates = [member.mask for member in members if member.mask is not None]
                if mask_candidates:
                    representative_mask = mask_candidates[0]
            best_cluster["representative"] = Detection(
                prompt_idx=-1,
                class_name=class_name,
                score=mean_score,
                box=mean_box,
                mask=representative_mask,
            )

        for cluster in clusters:
            votes = len(cluster["prompt_votes"])
            if votes < min_votes:
                continue
            consensus_clusters.append(
                {
                    "class_name": cluster["class_name"],
                    "votes": votes,
                    "score": float(np.mean([member.score for member in cluster["members"]])),
                    "box": cluster["representative"].box,
                    "mask": cluster["representative"].mask,
                    "members": cluster["members"],
                }
            )

    return consensus_clusters


def match_detections(
    detections: Sequence[Detection],
    references: Sequence[Dict[str, Any]],
    iou_threshold: float = 0.5,
    use_masks: bool = True,
) -> List[int]:
    matched_indices: List[int] = []
    used_references = set()

    ordered_detection_indices = sorted(range(len(detections)), key=lambda idx: detections[idx].score, reverse=True)

    for det_idx in ordered_detection_indices:
        det = detections[det_idx]
        best_ref_idx = None
        best_iou = -1.0

        for ref_idx, ref in enumerate(references):
            if ref_idx in used_references or ref["class_name"] != det.class_name:
                continue

            ref_det = Detection(
                prompt_idx=-1,
                class_name=ref["class_name"],
                score=float(ref["score"]),
                box=np.asarray(ref["box"], dtype=np.float32),
                mask=ref.get("mask"),
            )
            iou = detection_iou(det, ref_det, use_masks=use_masks)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_ref_idx = ref_idx

        if best_ref_idx is not None:
            matched_indices.append(det_idx)
            used_references.add(best_ref_idx)

    return matched_indices


def evaluate_prompt_sets_with_consensus(
    subset,
    prompt_sets: Sequence[Dict[str, Sequence[str]]],
    dino_processor,
    dino_model,
    sam_processor,
    sam_model,
    device: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    consensus_iou_threshold: float = 0.5,
    scoring_iou_threshold: float = 0.5,
    min_votes: int = 2,
    max_images: Optional[int] = None,
    max_detections_per_class: Optional[int] = None,
    return_masks: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    prompt_stats: Dict[int, Dict[str, float]] = {
        idx: {
            "consensus_matches": 0.0,
            "consensus_total": 0.0,
            "detections_total": 0.0,
            "unsupported_detections": 0.0,
            "images_with_any_detection": 0.0,
        }
        for idx in range(len(prompt_sets))
    }

    consensus_by_image: Dict[int, List[Dict[str, Any]]] = {}
    num_images = len(subset) if max_images is None else min(len(subset), max_images)

    for image_idx in range(num_images):
        sample = subset[image_idx]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample["image"]

        detections_by_prompt: Dict[int, List[Detection]] = {}
        all_detections: List[Detection] = []

        for prompt_idx, prompt_set in enumerate(prompt_sets):
            detections = run_prompt_set_on_image(
                image=image,
                prompt_idx=prompt_idx,
                prompt_set=prompt_set,
                dino_processor=dino_processor,
                dino_model=dino_model,
                sam_processor=sam_processor,
                sam_model=sam_model,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_detections_per_class=max_detections_per_class,
                return_masks=return_masks,
            )
            detections_by_prompt[prompt_idx] = detections
            all_detections.extend(detections)

            prompt_stats[prompt_idx]["detections_total"] += len(detections)
            if detections:
                prompt_stats[prompt_idx]["images_with_any_detection"] += 1

        consensus = build_consensus_clusters(
            detections=all_detections,
            min_votes=min_votes,
            iou_threshold=consensus_iou_threshold,
            use_masks=return_masks,
        )
        consensus_by_image[image_idx] = consensus

        for prompt_idx, detections in detections_by_prompt.items():
            matched = match_detections(
                detections=detections,
                references=consensus,
                iou_threshold=scoring_iou_threshold,
                use_masks=return_masks,
            )
            prompt_stats[prompt_idx]["consensus_matches"] += len(matched)
            prompt_stats[prompt_idx]["consensus_total"] += len(consensus)
            prompt_stats[prompt_idx]["unsupported_detections"] += max(0, len(detections) - len(matched))

    rows: List[Dict[str, Any]] = []
    for prompt_idx, stats in prompt_stats.items():
        consensus_total = max(1.0, stats["consensus_total"])
        detections_total = max(1.0, stats["detections_total"])
        support_precision = stats["consensus_matches"] / detections_total
        consensus_recall = stats["consensus_matches"] / consensus_total
        if support_precision + consensus_recall > 0:
            consensus_f1 = 2 * support_precision * consensus_recall / (support_precision + consensus_recall)
        else:
            consensus_f1 = 0.0

        rows.append(
            {
                "prompt_idx": prompt_idx,
                "car_prompts": ", ".join(prompt_sets[prompt_idx]["car_prompts"]),
                "pedestrian_prompts": ", ".join(prompt_sets[prompt_idx]["pedestrian_prompts"]),
                "detections_total": int(stats["detections_total"]),
                "consensus_matches": int(stats["consensus_matches"]),
                "consensus_total": int(stats["consensus_total"]),
                "support_precision": support_precision,
                "consensus_recall": consensus_recall,
                "consensus_f1": consensus_f1,
                "unsupported_rate": stats["unsupported_detections"] / detections_total,
                "images_with_any_detection": int(stats["images_with_any_detection"]),
            }
        )

    rows.sort(
        key=lambda row: (
            row["consensus_f1"],
            row["consensus_recall"],
            row["support_precision"],
        ),
        reverse=True,
    )
    return rows, consensus_by_image
