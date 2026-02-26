import argparse
import os
import time

import wandb
import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm
from PIL import Image

from src.models import DeTR, FasterRCNN, YOLOModel
from src.custom_datasets import KittiDataset, KittiDatasetUltralytics

from dotenv import load_dotenv
load_dotenv()


def main_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = args.model
    batch_size = args.batch_size
    threshold = args.threshold

    ds = KittiDataset('dataset/KITTI-MOTS', 'instances_txt', 'training',
                      'src/custom_datasets/val.seqmap')

    if model_type == "fasterrcnn":
        variant = args.variant
        model_name = f"{model_type} ({variant})"
        detector = FasterRCNN(
            variant=variant, threshold=threshold, device=device)
        detector.set_eval_mode()
        coco_categories = detector.categories
        wandb.init(
            project="C5-Week1",
            entity="c5-team2",
            name=f"Inference-{model_type}-{variant}",
            config={
                "model": model_type,
                "variant": variant,
                "batch_size": batch_size,
            }
        )

    elif model_type == "detr":
        model_name = f"{model_type} ({variant})"
        detector = DeTR(variant=variant, threshold=threshold, device=device)
        coco_categories = detector.model.config.id2label
        wandb.init(
            project="C5-Week1",
            entity="c5-team2",
            name=f"Inference-{model_type}",
            config={
                "model": model_type,
                "batch_size": batch_size,
            }
        )

    elif model_type == "yolo":
        variant = args.variant
        model_name = f"{variant}"
        detector = YOLOModel(model=variant, threshold=threshold, device=device)
        coco_categories = detector.model.names
        ds = KittiDatasetUltralytics(
            'dataset/KITTI-MOTS', 'instances_txt', 'training',
            'src/custom_datasets/val.seqmap')
        wandb.init(
            project="C5-Week1",
            entity="c5-team2",
            name=f"Inference-{model_type}-{variant}",
            config={
                "model": model_type,
                "batch_size": batch_size,
            }
        )

    else:
        raise ValueError(f"Unknown model {model_type}.")

    loader = DataLoader(ds, batch_size=batch_size,
                        collate_fn=lambda x: list(x))

    total_time = 0
    total_images = 0

    for i, images in enumerate(tqdm(loader, desc="Running Inference")):

        # For task g
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_batch = time.perf_counter()

        # list of pairs image and target
        if (isinstance(images, list) and len(images) > 0
                and isinstance(images[0], tuple)):
            images = [im[0] for im in images]

        preds = detector.inference(images)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_batch = time.perf_counter()

        total_time += (end_batch - start_batch)
        total_images += len(images)

        # Visualizing results for the first few batches
        if i == 130 // batch_size:
            for j, (img, pred) in enumerate(zip(images, preds)):
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                if isinstance(img, Image.Image):
                    img = ToTensor()(img)

                img_uint8 = (img * 255).to(torch.uint8)

                boxes = pred['boxes']
                labels = pred['labels']
                scores = pred['scores']

                label_names = [
                    f"{coco_categories[l.item()]}: {s:.2f}" for l, s in zip(labels, scores)]

                if len(boxes) > 0:
                    result_img = draw_bounding_boxes(
                        img_uint8,
                        boxes=boxes,
                        labels=label_names,
                        colors="yellow",
                        width=3
                    )

                    os.makedirs("results/task_c/", exist_ok=True)
                    save_path = f"results/task_c/{model_type}_batch{i}_img{j}.png"
                    F.to_pil_image(result_img).save(save_path)

    avg_time_per_img = total_time / total_images
    fps = 1 / avg_time_per_img  # frames per second

    wandb.log({
        "performance/avg_time_per_img": avg_time_per_img,
        "performance/fps": fps,
        "performance/total_time": total_time
    })

    print(f"Model: {model_name}")
    print(f"Total Parameters: {sum(p.numel() for p in detector.model.parameters()):,}")
    print(f"Total Images Processed: {total_images}")
    print(f"Total Inference Time: {total_time:.2f}s")
    print(f"Average Time per Image: {avg_time_per_img:.4f}s")
    print(f"Inference Speed: {fps:.2f} FPS")
    print(f"Inference finished. Qualitative results saved in 'results/task_c'.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fastrrcnn", choices=[
                        "fasterrcnn", "detr", "yolo"], help="Main model to use on inference. fasterrcnn uses Torchvision, detr uses HuggingFace and yolo uses Ultralytics. Defaults to fasterrcnn.")
    parser.add_argument("--variant", type=str, default="resnet50_fpn_v2",
                        help="Variant of the model to use on inference. Defaults to resnet50_fpn_v2.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference. Defaults to 16.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for the score of each bounding box. Defaults to 0.5.")
    args = parser.parse_args()

    main_inference(args)
