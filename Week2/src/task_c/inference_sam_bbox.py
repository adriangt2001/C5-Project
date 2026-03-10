import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from tqdm import tqdm
import io
import matplotlib.pyplot as plt
from PIL import Image
import wandb
import numpy as np
import time
import random

from src.utils.kitti_dataset_motsio import KittiDataset
from src.utils.metrics import InstanceSegmentationMetrics, parse_segmentation_metrics
from src.utils.visualizations import show_mask, show_box

def make_visualization(image_tensor, gt_masks, pred_masks, boxes=None, title=""):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GT
    axes[0].imshow(image_np)
    for mask in gt_masks:
        show_mask(mask.cpu().numpy(), axes[0], random_color=True)
    axes[0].set_title("GT Masks")
    axes[0].axis("off")

    # Pred
    axes[1].imshow(image_np)
    for mask in pred_masks:
        show_mask(mask.cpu().numpy(), axes[1], random_color=True)

    if boxes is not None:
        for box in boxes:
            show_box(box, axes[1])

    axes[1].set_title("Predicted Masks")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

@torch.no_grad()
def run_sam_on_boxes(image_tensor, boxes, processor, sam_model, device):
    """
    image_tensor: [3,H,W] in [0,1]
    boxes: np.ndarray [N,4] in xyxy
    """
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

def main_task_c(args):
    csv_path = args.csv_path
    image_folder = args.image_folder
    annotations_folder = args.annotations_folder
    seqmap_file = args.seqmap_file
    score_mode = args.score_mode
    run_name = args.run_name
    batch_size = args.batch_size
    max_visualizations = args.max_visualizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="C5-Week2",
        name=run_name,
        config={
            "csv_path": csv_path,
            "sam_model": "facebook/sam-vit-base",
            "score_mode": score_mode,
        },
    )

    df = pd.read_csv(csv_path)

    # Car and pedestrian only, in case there is an error with the csv
    df = df[df["label_id"].isin([1, 2])].copy()

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

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()

    metric = InstanceSegmentationMetrics(class_metrics=True)
    metric.reset()

    total_time = 0.0
    total_images = 0
    logged_visuals = 0

    for images, targets in tqdm(loader, desc="Task C inference"):
        for image, target in zip(images, targets):
            image_id = int(target["image_id"])

            detections = df[df["image_id"] == image_id]

            boxes = detections[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
            labels = detections["label_id"].to_numpy(dtype=np.int64) 
            scores = detections["score"].to_numpy(dtype=np.float32)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            pred_masks, sam_scores = run_sam_on_boxes(
                image_tensor=image,
                boxes=boxes,
                processor=processor,
                sam_model=sam_model,
                device=device,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_time += end - start
            total_images += 1

            if len(boxes) == 0:
                pred = {
                    "masks": torch.zeros(
                        (0, image.shape[1], image.shape[2]),
                        dtype=torch.bool,
                    ),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            else:
                det_scores = torch.tensor(scores, dtype=torch.float32)

                if score_mode == "detector":
                    final_scores = det_scores
                elif score_mode == "sam":
                    final_scores = sam_scores
                elif score_mode == "product":
                    final_scores = det_scores * sam_scores
                else:
                    raise ValueError(f"Unknown score_mode: {score_mode}")

                pred = {
                    "masks": pred_masks,
                    "scores": final_scores,
                    "labels": torch.tensor(labels, dtype=torch.int64),
                }

            gt = {
                "masks": target["masks"].bool(),
                "labels": target["labels"].long(),
            }

            metric.update([pred], [gt])

            if image_id in selected_visualization_ids:
                vis_img = make_visualization(
                    image_tensor=image,
                    gt_masks=gt["masks"],
                    pred_masks=pred["masks"],
                    boxes=boxes if len(boxes) > 0 else None,
                    title=f"image_id={image_id}",
                )

                wandb.log({
                    f"qualitative/image_{image_id}": wandb.Image(
                        vis_img,
                        caption=f"image_id={image_id}",
                    )
                })
                logged_visuals += 1

    results = metric.compute()
    metrics = parse_segmentation_metrics(
        results,
        id2label={
            1: "car",
            2: "pedestrian"
        }
    )

    avg_time_per_img = total_time / max(total_images, 1)
    fps = 1.0 / avg_time_per_img if avg_time_per_img > 0 else 0.0

    wandb.log({
        **{f"metrics/{k}": v for k, v in metrics.items()},
        "performance/total_images": total_images,
        "performance/total_inference_time": total_time,
        "performance/avg_time_per_img": avg_time_per_img,
        "performance/fps": fps,
    })

    print("\n=== Task C metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== Performance ===")
    print(f"Total images: {total_images}")
    print(f"Total inference time: {total_time:.4f}s")
    print(f"Average time per image: {avg_time_per_img:.6f}s")
    print(f"FPS: {fps:.2f}")

    wandb.finish()