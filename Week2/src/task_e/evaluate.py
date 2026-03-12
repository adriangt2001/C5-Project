import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK2_SRC = PROJECT_ROOT / "Week2" / "src"

if str(WEEK2_SRC) not in sys.path:
    sys.path.append(str(WEEK2_SRC))

from utils.kitti_dataset_motsio2 import KittiDataset
from utils.metrics import InstanceSegmentationMetrics, parse_segmentation_metrics


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_argparser():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SAM checkpoints on KITTI-MOTS val set.")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "data" / "KITTI-MOTS" / "training",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "data" / "KITTI-MOTS" / "instances_txt",
    )
    parser.add_argument(
        "--val-seqmap",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "src" / "utils" / "val.seqmap",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "task_e" / "sam_vit_b_01ec64.pth",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=None,
        help="One or more fine-tuned checkpoints to evaluate. If omitted, evaluates all `sam_task_e_*.pth` in task_e/checkpoints.",
    )
    parser.add_argument("--model-type", type=str, default="vit_b")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Evaluate the original Meta SAM checkpoint used to initialize fine-tuning.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "task_e" / "evaluation_metrics.json",
    )
    parser.add_argument("--visualize", dest="visualize", action="store_true")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false")
    parser.set_defaults(visualize=False)
    parser.add_argument(
        "--visualize-count",
        type=int,
        default=5,
        help="How many evaluation samples to save as qualitative visualizations.",
    )
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "task_e" / "evaluation_visualizations",
    )

    return parser


def mask_to_box(mask: torch.Tensor) -> torch.Tensor | None:
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0 or xs.numel() == 0:
        return None

    return torch.tensor(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=torch.float32,
    )


def resolve_eval_checkpoints(checkpoints: list[Path] | None) -> list[Path]:
    if checkpoints:
        missing = [path for path in checkpoints if not path.exists()]
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Checkpoint(s) not found: {missing_str}")
        return [path.resolve() for path in checkpoints]

    checkpoint_dir = PROJECT_ROOT / "Week2" / "task_e" / "checkpoints"
    discovered = sorted(checkpoint_dir.glob("sam_task_e_*.pth"))
    if not discovered:
        raise FileNotFoundError(
            f"No fine-tuned checkpoints found in {checkpoint_dir}. "
            "Pass `--checkpoints /path/to/model.pth` explicitly."
        )
    return discovered


def build_dataset_and_loader(args):
    dataset = KittiDataset(
        str(args.data_root),
        str(args.annotations_root),
        str(args.val_seqmap),
        only_mask=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    return dataset, loader


def prepare_image(image: torch.Tensor, sam_model, resize_transform: ResizeLongestSide, device: torch.device):
    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    resized_image = resize_transform.apply_image(image_np)
    resized_size = resized_image.shape[:2]
    resized_tensor = torch.as_tensor(resized_image, device=device).permute(2, 0, 1).contiguous()
    input_image = sam_model.preprocess(resized_tensor.unsqueeze(0))
    return input_image, image_np.shape[:2], resized_size


def overlay_mask(ax, mask: np.ndarray, color: tuple[float, float, float, float]):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)


def draw_box(ax, box: np.ndarray, color: str = "lime"):
    x0, y0, x1, y1 = box
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            edgecolor=color,
            facecolor=(0, 0, 0, 0),
            linewidth=2,
        )
    )


def save_visualization(
    image: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_boxes: list[torch.Tensor],
    pred_masks: torch.Tensor,
    pred_scores: torch.Tensor,
    save_path: Path,
):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    gt_masks_np = gt_masks.cpu().numpy()
    pred_masks_np = pred_masks.cpu().numpy()
    pred_scores_np = pred_scores.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image_np)
    axes[0].set_title("GT box prompts")
    axes[0].axis("off")
    for box in gt_boxes:
        draw_box(axes[0], box.cpu().numpy())

    axes[1].imshow(image_np)
    axes[1].set_title("SAM predicted masks")
    axes[1].axis("off")

    for idx, mask in enumerate(pred_masks_np):
        overlay_mask(axes[1], mask, (0.13, 0.8, 0.33, 0.45))
        if idx < len(gt_boxes):
            box = gt_boxes[idx].cpu().numpy()
            axes[1].text(
                box[0],
                max(0, box[1] - 5),
                f"{pred_scores_np[idx]:.3f}",
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
            )

    if gt_masks_np.shape[0] > 0:
        for mask in gt_masks_np:
            overlay_mask(axes[0], mask, (0.12, 0.56, 1.0, 0.20))

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def predict_image_instances(model, image: torch.Tensor, target: dict, device: torch.device, resize_transform):
    gt_masks = target["masks"]
    gt_labels = target["labels"].long()
    height, width = image.shape[1:]

    if gt_masks.numel() == 0 or gt_masks.shape[0] == 0:
        pred = {
            "masks": torch.zeros((0, height, width), dtype=torch.bool),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
        gt = {
            "masks": gt_masks.bool(),
            "labels": gt_labels,
        }
        debug = {
            "boxes": [],
            "pred_scores": pred["scores"],
        }
        return pred, gt, debug

    input_image, original_size, resized_size = prepare_image(image, model, resize_transform, device)
    image_embeddings = model.image_encoder(input_image)

    pred_masks = []
    pred_scores = []
    pred_labels = []
    gt_boxes = []

    for mask, label in zip(gt_masks, gt_labels):
        box = mask_to_box(mask)
        if box is None:
            continue

        gt_boxes.append(box)
        resized_box = resize_transform.apply_boxes_torch(box.unsqueeze(0), original_size).to(device)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=resized_box,
            masks=None,
        )
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upsampled_masks = model.postprocess_masks(
            low_res_masks,
            input_size=resized_size,
            original_size=original_size,
        )

        pred_masks.append((torch.sigmoid(upsampled_masks[:, 0]) > 0.5).cpu())
        pred_scores.append(torch.sigmoid(iou_predictions[:, 0]).cpu())
        pred_labels.append(label.view(1).cpu())

    if pred_masks:
        pred = {
            "masks": torch.cat(pred_masks, dim=0).bool(),
            "scores": torch.cat(pred_scores, dim=0).float(),
            "labels": torch.cat(pred_labels, dim=0).long(),
        }
    else:
        pred = {
            "masks": torch.zeros((0, height, width), dtype=torch.bool),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    gt = {
        "masks": gt_masks.bool(),
        "labels": gt_labels,
    }
    debug = {
        "boxes": gt_boxes,
        "pred_scores": pred["scores"],
    }
    return pred, gt, debug


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, args, loader, device: torch.device):
    model = sam_model_registry[args.model_type](checkpoint=str(args.sam_checkpoint))
    if checkpoint_path.resolve() != args.sam_checkpoint.resolve():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    resize_transform = ResizeLongestSide(model.image_encoder.img_size)

    metric = InstanceSegmentationMetrics(class_metrics=True)
    metric.reset()
    saved_visualizations = 0

    for images, targets in tqdm(loader, desc=f"eval {checkpoint_path.name}"):
        for image, target in zip(images, targets):
            pred, gt, debug = predict_image_instances(
                model=model,
                image=image.to(device),
                target=target,
                device=device,
                resize_transform=resize_transform,
            )
            metric.update([pred], [gt])

            if args.visualize and saved_visualizations < args.visualize_count:
                save_name = f"{checkpoint_path.stem}_sample_{saved_visualizations:03d}.png"
                save_visualization(
                    image=image,
                    gt_masks=gt["masks"],
                    gt_boxes=debug["boxes"],
                    pred_masks=pred["masks"],
                    pred_scores=debug["pred_scores"],
                    save_path=args.visualize_dir / save_name,
                )
                saved_visualizations += 1

    results = metric.compute()
    parsed = parse_segmentation_metrics(
        results,
        id2label={1: "car", 2: "pedestrian"},
    )
    return parsed


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(
            f"Base SAM checkpoint not found at {args.sam_checkpoint}. "
            "This file is required to rebuild the model before loading fine-tuned weights."
        )

    if args.pretrained:
        checkpoints = [args.sam_checkpoint.resolve()]
    else:
        checkpoints = resolve_eval_checkpoints(args.checkpoints)
    _, loader = build_dataset_and_loader(args)

    all_metrics = {}
    for checkpoint_path in checkpoints:
        metrics = evaluate_checkpoint(checkpoint_path, args, loader, device)
        all_metrics[checkpoint_path.name] = metrics

        print(f"\n=== {checkpoint_path.name} ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)

    print(f"\nSaved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
