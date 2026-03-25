from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import wandb
import time

from .dataset import load_annotations, VizWizCaptionDataset, collate_fn
from src.utils import compute_metrics

def load_model_and_processor(args, device):
    if args.model_type == "vit-gpt2":
        model = VisionEncoderDecoderModel.from_pretrained(args.model_name, use_safetensors=True)
        processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    elif args.model_type == "blip":
        processor = BlipProcessor.from_pretrained(args.model_name, use_fast=False)
        model = BlipForConditionalGeneration.from_pretrained(args.model_name, use_safetensors=True)
        tokenizer = None  # blip uses the processor for everything
        
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.to(device)
    model.eval()
    return model, processor, tokenizer

def generate_captions(model, processor, tokenizer, batch, args, device):
    pixel_values = batch["pixel_values"].to(device)

    if args.model_type == "vit-gpt2":
        generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,)
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    else:
        generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,)
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
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

    model, processor, tokenizer = load_model_and_processor(args, device)

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
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            predictions = generate_captions(model, processor, tokenizer, batch, args, device)

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

    total_time = time.time() - start_time
    fps = len(results) / total_time 

    print(f"Total time: {total_time:.2f}s")
    print(f"FPS: {fps:.2f} imgs/s")

    print("Computing metrics...")
    metrics = compute_metrics(results)

    if wandb_cfg["enabled"]:
        wandb.log({
            "metrics/bleu1": metrics["bleu1"],
            "metrics/bleu2": metrics["bleu2"],
            "metrics/rougeL": metrics["rougeL"],
            "metrics/meteor": metrics["meteor"],
            "performance/fps": fps,
            "performance/total_time_s": total_time,
            "performance/num_images": len(results),
        })
        wandb.finish()
        
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "metrics": metrics,
        "results": results,
    }, indent=2))
    print(f"Saved {len(results)} predictions → {output_path}")