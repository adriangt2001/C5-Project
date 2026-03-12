import io
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor, pipeline

from src.utils.kitti_dataset_motsio import KittiDataset
from src.utils.metrics import SemanticSegmentationMetrics
from src.utils.visualizations import show_box, show_semantic_mask, plot_confusion_matrix

PROMPT_MODES = {
    "baseline": {
        "car_prompts": ["car"],
        "pedestrian_prompts": ["person"],
    },
    "synonyms": {
        "car_prompts": ["car", "automobile", "vehicle"],
        "pedestrian_prompts": ["person", "pedestrian", "human"],
    },
    "context": {
        "car_prompts": ["car on road", "road vehicle"],
        "pedestrian_prompts": ["person walking", "walking pedestrian"],
    },
}

CAR_KEYWORDS = ["car", "automobile", "vehicle", "road vehicle", "car on road"]
PEDESTRIAN_KEYWORDS = ["person", "pedestrian", "human", "person walking", "walking pedestrian"]

def make_visualization(image_tensor, gt_semantic, pred_semantic, boxes=None, title=None):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image_np)
    show_semantic_mask(gt_semantic.cpu().numpy(), axes[0])
    axes[0].set_title("GT Semantic")
    axes[0].axis("off")

    axes[1].imshow(image_np)
    show_semantic_mask(pred_semantic.cpu().numpy(), axes[1])

    if boxes is not None:
        for box in boxes:
            show_box(box, axes[1])

    axes[1].set_title("Pred Semantic")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

@torch.no_grad
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
    )[0]  # [N, 1, H, W]

    masks = masks[:, 0] > 0
    sam_scores = outputs.iou_scores[0][:, 0].detach().cpu().float()

    return masks.bool(), sam_scores

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def instances_to_semantic_mask(masks, labels, scores, height, width):
    semantic = torch.zeros((height, width), dtype=torch.long)
    occupied = torch.zeros((height, width), dtype=torch.bool)

    if masks.shape[0] == 0:
        return semantic

    order = torch.argsort(scores, descending=True)

    for idx in order:
        current_mask = masks[idx].bool() & (~occupied)
        semantic[current_mask] = labels[idx].long()
        occupied[current_mask] = True

    return semantic

@torch.no_grad
def run_groundingdino_on_image(
    image_tensor,
    prompt_mode,
    box_threshold,
    object_detector,
):
    mode = PROMPT_MODES[prompt_mode]
    all_prompts = mode["car_prompts"] + mode["pedestrian_prompts"]

    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)

    candidate_labels = [p if p.endswith(".") else p + "." for p in all_prompts]

    results = object_detector(
        image_pil,
        candidate_labels=candidate_labels,
        threshold=box_threshold,
    )

    if len(results) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    boxes, labels, scores = [], [], []
    for r in results:
        label_str = r["label"].replace(".", "").strip().lower()

        if label_str in CAR_KEYWORDS:
            class_id = 1
        elif label_str in PEDESTRIAN_KEYWORDS:
            class_id = 2
        else:
            continue

        b = r["box"]
        boxes.append([b["xmin"], b["ymin"], b["xmax"], b["ymax"]])
        labels.append(class_id)
        scores.append(r["score"])

    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.array(boxes, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(scores, dtype=np.float32),
    )


def main_task_h(args):
    image_folder = args.image_folder
    annotations_folder = args.annotations_folder
    seqmap_file = args.seqmap_file
    prompt_mode = args.prompt_mode
    score_mode = args.score_mode
    run_name = args.run_name
    batch_size = args.batch_size
    max_visualizations = args.max_visualizations
    box_threshold = args.box_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="C5-Week2",
        name=run_name,
        config={
            "sam_model": "facebook/sam-vit-base",
            "dino_model": "IDEA-Research/grounding-dino-tiny",
            "prompt_mode": prompt_mode,
            "score_mode": score_mode,
            "box_threshold": box_threshold,
        },
    )

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

    random.seed(42)
    all_image_ids = [int(dataset[i][1]["image_id"]) for i in range(len(dataset))]
    selected_visualization_ids = set(
        random.sample(
            all_image_ids,
            k=min(max_visualizations, len(all_image_ids))
        )
    )

    wandb.config.update({
        "visualization_ids": sorted(list(selected_visualization_ids))
    })

    # Inicializar GroundingDINO
    print("Loading GroundingDINO...")
    dino = pipeline(
        model="IDEA-Research/grounding-dino-tiny",
        task="zero-shot-object-detection",
        device=device,
    )

    # Inicializar SAM
    print("Loading SAM...")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    metric = SemanticSegmentationMetrics(num_classes=3)
    metric.reset()

    total_time = 0.0
    total_images = 0

    for images, targets in tqdm(loader, desc="Grounded-SAM semantic inference"):
        for image, target in zip(images, targets):
            image_id = int(target["image_id"])
            h, w = image.shape[1:]

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            # GroundingDINO: text prompts -> boxes
            boxes, labels_np, det_scores_np = run_groundingdino_on_image(
                image_tensor=image,
                prompt_mode=prompt_mode,
                box_threshold=box_threshold,
                object_detector=dino,
            )

            # SAM: boxes -> masks
            pred_masks, sam_scores = run_sam_on_boxes(
                image_tensor=image,
                boxes=boxes,
                processor=processor,
                sam_model=sam_model,
                device=device,
            )

            # Score fusion
            if len(boxes) == 0:
                final_scores = torch.zeros((0,), dtype=torch.float32)
                pred_labels = torch.zeros((0,), dtype=torch.long)
            else:
                det_scores = torch.tensor(det_scores_np, dtype=torch.float32)
                pred_labels = torch.tensor(labels_np, dtype=torch.long)

                if score_mode == "detector":
                    final_scores = det_scores
                elif score_mode == "sam":
                    final_scores = sam_scores
                elif score_mode == "product":
                    final_scores = det_scores * sam_scores
                else:
                    raise ValueError(f"Unknown score_mode: {score_mode}")

            # Instances -> semantic mask
            pred_semantic = instances_to_semantic_mask(
                masks=pred_masks,
                labels=pred_labels,
                scores=final_scores,
                height=h,
                width=w,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_time += end - start
            total_images += 1

            gt_semantic = target["semantic_mask"].long()
            metric.update(pred_semantic, gt_semantic)

            if image_id in selected_visualization_ids:
                vis_img = make_visualization(
                    image_tensor=image,
                    gt_semantic=gt_semantic,
                    pred_semantic=pred_semantic,
                    boxes=boxes if len(boxes) > 0 else None,
                )
                wandb.log({
                    f"qualitative/image_{image_id}": wandb.Image(
                        vis_img,
                        caption=f"image_id={image_id}",
                    )
                })

    results = metric.compute()

    iou_per_class = results["iou_per_class"]
    cm = results["confusion_matrix"]
    class_names = ["background", "car", "pedestrian"]

    avg_time_per_img = total_time / max(total_images, 1)
    fps = 1.0 / avg_time_per_img if avg_time_per_img > 0 else 0.0

    wandb.log({
        "metrics/pixel_accuracy": results["pixel_accuracy"],
        "metrics/mean_accuracy": results["mean_accuracy"],
        "metrics/miou": results["miou"],
        "metrics/iou_car": iou_per_class[1].item(),
        "metrics/iou_pedestrian": iou_per_class[2].item(),
        "metrics/confusion_matrix": wandb.Image(plot_confusion_matrix(cm, class_names)),

        "performance/total_images": total_images,
        "performance/total_inference_time": total_time,
        "performance/avg_time_per_img": avg_time_per_img,
        "performance/fps": fps,
    })

    print("\n=== Semantic Segmentation Metrics ===")
    print(f"Pixel Accuracy:  {results['pixel_accuracy']:.4f}")
    print(f"Mean Accuracy:   {results['mean_accuracy']:.4f}")
    print(f"mIoU:            {results['miou']:.4f}")
    print(f"IoU car:         {iou_per_class[1].item():.4f}")
    print(f"IoU pedestrian:  {iou_per_class[2].item():.4f}")

    print("\n=== Performance ===")
    print(f"Total images:         {total_images}")
    print(f"Total inference time: {total_time:.4f}s")
    print(f"Avg time per image:   {avg_time_per_img:.6f}s")
    print(f"FPS:                  {fps:.2f}")

wandb.finish()