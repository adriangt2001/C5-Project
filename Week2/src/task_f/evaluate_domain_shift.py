import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ISICDataset(Dataset):
    def __init__(self, image_dir: str | Path, mask_dir: str | Path):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        image_paths = sorted(self.image_dir.glob("*.jpg"))
        self.samples = []
        print(f"Found {len(image_paths)} images in {self.image_dir}")

        for image_path in image_paths:
            stem = image_path.stem
            mask_path = self.mask_dir / f"{stem}_segmentation.png"
            if mask_path.exists():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No paired ISIC samples found in {self.image_dir} with masks in {self.mask_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) > 0

        return {
            "image": torch.tensor(image).permute(2, 0, 1).float() / 255.0,
            "mask": torch.tensor(mask, dtype=torch.bool),
            "image_id": idx,
            "image_path": str(image_path),
        }


def collate_fn(batch):
    return batch


def mask_to_box(mask: torch.Tensor) -> torch.Tensor | None:
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    return torch.tensor(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=torch.float32,
    )


def prepare_image(image: torch.Tensor, sam_model, resize_transform: ResizeLongestSide, device: torch.device):
    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    resized_image = resize_transform.apply_image(image_np)
    resized_size = resized_image.shape[:2]
    resized_tensor = torch.as_tensor(resized_image, device=device).permute(2, 0, 1).contiguous()
    input_image = sam_model.preprocess(resized_tensor.unsqueeze(0))
    return input_image, image_np.shape[:2], resized_size


def rle_encode_binary_mask(mask: np.ndarray):
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def build_coco_gt(dataset: ISICDataset):
    images = []
    annotations = []

    ann_id = 1
    for sample in dataset:
        mask = sample["mask"].numpy()
        h, w = mask.shape
        image_id = int(sample["image_id"])
        bbox_tensor = mask_to_box(sample["mask"])
        area = float(mask.sum())

        images.append(
            {
                "id": image_id,
                "file_name": Path(sample["image_path"]).name,
                "height": h,
                "width": w,
            }
        )

        if bbox_tensor is None:
            continue

        x0, y0, x1, y1 = bbox_tensor.tolist()
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": area,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "segmentation": rle_encode_binary_mask(mask),
            }
        )
        ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "lesion"}],
    }


def build_model(base_checkpoint: Path, model_type: str, finetuned_checkpoint: Path | None, device: torch.device):
    model = sam_model_registry[model_type](checkpoint=str(base_checkpoint))
    if finetuned_checkpoint is not None:
        state_dict = torch.load(finetuned_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_sample(model, sample: dict, device: torch.device):
    image = sample["image"].to(device)
    gt_mask = sample["mask"]
    image_id = int(sample["image_id"])
    bbox = mask_to_box(gt_mask)

    if bbox is None:
        return None

    resize_transform = ResizeLongestSide(model.image_encoder.img_size)
    input_image, original_size, resized_size = prepare_image(image, model, resize_transform, device)
    image_embeddings = model.image_encoder(input_image)

    resized_box = resize_transform.apply_boxes_torch(bbox.unsqueeze(0), original_size).to(device)
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

    pred_masks = model.postprocess_masks(
        low_res_masks,
        input_size=resized_size,
        original_size=original_size,
    )

    pred_mask = (torch.sigmoid(pred_masks[:, 0]) > 0.5)[0].cpu().numpy()
    score = float(torch.sigmoid(iou_predictions[:, 0])[0].item())

    return {
        "image_id": image_id,
        "category_id": 1,
        "score": score,
        "segmentation": rle_encode_binary_mask(pred_mask),
    }, bbox.cpu().numpy(), pred_mask


def overlay_mask(ax, mask: np.ndarray, color):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)


def draw_box(ax, box: np.ndarray, color="lime"):
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


def make_visualization(image: torch.Tensor, gt_mask: torch.Tensor, bbox: np.ndarray, pred_mask: np.ndarray, title: str):
    image_np = image.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image_np)
    overlay_mask(axes[0], gt_mask.numpy(), (0.12, 0.56, 1.0, 0.35))
    draw_box(axes[0], bbox, color="lime")
    axes[0].set_title("GT mask + prompt box")
    axes[0].axis("off")

    axes[1].imshow(image_np)
    overlay_mask(axes[1], pred_mask, (0.13, 0.8, 0.33, 0.35))
    draw_box(axes[1], bbox, color="lime")
    axes[1].set_title("SAM prediction")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def coco_eval(gt_dict, predictions):
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "map": round(float(stats[0]), 4),
        "map_50": round(float(stats[1]), 4),
        "map_75": round(float(stats[2]), 4),
        "map_small": round(float(stats[3]), 4),
        "map_medium": round(float(stats[4]), 4),
        "map_large": round(float(stats[5]), 4),
        "mar_1": round(float(stats[6]), 4),
        "mar_10": round(float(stats[7]), 4),
        "mar_100": round(float(stats[8]), 4),
        "mar_small": round(float(stats[9]), 4),
        "mar_medium": round(float(stats[10]), 4),
        "mar_large": round(float(stats[11]), 4),
    }


def evaluate_model(model_name: str, model, loader, device: torch.device, visualize_count: int, visualize_dir: Path):
    predictions = []
    saved = 0

    for batch in tqdm(loader, desc=f"task_f {model_name}"):
        for sample in batch:
            result = predict_sample(model, sample, device)
            if result is None:
                continue

            prediction, bbox, pred_mask = result
            predictions.append(prediction)

            if saved < visualize_count:
                vis = make_visualization(
                    image=sample["image"],
                    gt_mask=sample["mask"],
                    bbox=bbox,
                    pred_mask=pred_mask,
                    title=f"{model_name} | image_id={sample['image_id']}",
                )
                visualize_dir.mkdir(parents=True, exist_ok=True)
                vis.save(visualize_dir / f"{model_name}_{int(sample['image_id']):04d}.png")
                saved += 1

    return predictions


def main_task_f(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ISICDataset(args.image_dir, args.mask_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    gt_dict = build_coco_gt(dataset)
    base_checkpoint = Path(args.sam_checkpoint)
    finetuned_checkpoint = Path(args.finetuned_checkpoint)

    pretrained_model = build_model(
        base_checkpoint=base_checkpoint,
        model_type=args.model_type,
        finetuned_checkpoint=None,
        device=device,
    )
    finetuned_model = build_model(
        base_checkpoint=base_checkpoint,
        model_type=args.model_type,
        finetuned_checkpoint=finetuned_checkpoint,
        device=device,
    )

    output_dir = Path(args.output_dir)
    pretrained_preds = evaluate_model(
        model_name="pretrained",
        model=pretrained_model,
        loader=loader,
        device=device,
        visualize_count=args.visualize_count,
        visualize_dir=output_dir / "visualizations" / "pretrained",
    )
    finetuned_preds = evaluate_model(
        model_name="finetuned",
        model=finetuned_model,
        loader=loader,
        device=device,
        visualize_count=args.visualize_count,
        visualize_dir=output_dir / "visualizations" / "finetuned",
    )

    metrics = {
        "pretrained": coco_eval(gt_dict, pretrained_preds),
        "finetuned_kitti_mots": coco_eval(gt_dict, finetuned_preds),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "task_f_metrics.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"\nSaved Task F metrics to {output_json}")
    for model_name, model_metrics in metrics.items():
        print(f"\n=== {model_name} ===")
        for key, value in model_metrics.items():
            print(f"{key}: {value}")
