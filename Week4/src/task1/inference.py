from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
from tqdm import tqdm
import wandb

from .dataset import load_annotations, VizWizCaptionDataset, collate_fn
from src.utils import compute_metrics

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

    # Had to put use_safetensors=True because of error with pytorch version 
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name, use_safetensors=True)

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.to(device)
    model.eval()

    # Using AutoImageProcessor and AutoTokenizer to be more general but they can be changed
    processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_dir = Path(args.data_dir)
    samples = load_annotations(data_dir / "annotations" / "val.json") # Our test set

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
            pixel_values = batch["pixel_values"].to(device)

            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

            predictions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

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