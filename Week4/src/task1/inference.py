from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .dataset import load_annotations, VizWizCaptionDataset, collate_fn
from src.utils import compute_metrics
from src.task1.models import load_model_and_processor


def generate_captions(model, processor, tokenizer, batch, args, device):
    pixel_values = batch["pixel_values"].to(device)

    if args.model_type == "vit-gpt2":
        generated_ids = model.generate(
            pixel_values=pixel_values, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,)
        predictions = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
    else:
        generated_ids = model.generate(
            pixel_values=pixel_values, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,)
        predictions = processor.batch_decode(
            generated_ids, skip_special_tokens=True)

    return predictions


def run_inference(args):

    wandb_cfg = args.wandb
    if wandb_cfg["enabled"]:
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=wandb_cfg["name"],
            config=vars(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, processor, tokenizer = load_model_and_processor(
        args.model_type, args.model_name, device, mode="inference")

    data_dir = Path(args.data_dir)
    samples = load_annotations(
        data_dir / "annotations" / "val.json")  # Our test set

    dataset = VizWizCaptionDataset(
        data_dir=data_dir,
        samples=samples,
        processor=processor,
        training=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            predictions = generate_captions(
                model, processor, tokenizer, batch, args, device)

            for pred, refs, fname, img_id in zip(
                predictions,
                batch["references"],
                batch["file_names"],
                batch["image_ids"],
            ):
                results.append({
                    "image_id": img_id,
                    "file_name": fname,
                    "prediction": pred.strip(),
                    "references": refs,
                })

    print("Computing metrics...")
    metrics = compute_metrics(results)
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if wandb_cfg["enabled"]:
        wandb.log(metrics)
        wandb.finish()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "metrics": metrics,
        "results": results,
    }, indent=2))
    print(f"Saved {len(results)} predictions → {output_path}")
