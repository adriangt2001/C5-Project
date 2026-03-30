from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Sequence, Dict
import time

from src.utils.dataset import load_annotations, VizWizCaptionDataset, collate_fn
from src.utils import compute_metrics
from src.utils.models import load_model_and_processor


def generate_captions(model, processor, tokenizer, batch, args, device):

    if args.model_type == "vit-gpt2":
        pixel_values = batch["pixel_values"].to(device)
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if args.model_type == "qwen3.5_9b":
        prompts = [
            "Describe this image briefly in a single sentence."
        ] * args.batch_size
        list_of_images = batch["images"]

        texts = [
            processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": p}],
                    }
                ],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]

        inputs = processor(
            text=texts, images=list_of_images, return_tensors="pt", padding=True
        ).to(model.device, dtype=torch.bfloat16)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        predictions = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
    else:
        pixel_values = batch["pixel_values"].to(device)
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
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

    model, processor, tokenizer = load_model_and_processor(
        args.model_type, args.model_name, device, mode="inference"
    )

    is_llm = args.model_type == "qwen3.5_9b"
    data_dir = Path(args.data_dir)
    samples = load_annotations(data_dir / "annotations" / "val.json")  # Our test set

    dataset = VizWizCaptionDataset(
        data_dir=data_dir,
        samples=samples,
        processor=processor,
        is_llm=is_llm,
        training=False,
    )

    if is_llm:

        def collate_fn(batch: Sequence[Dict]) -> Dict:
            images = [item["image"] for item in batch]

            return {
                "images": images,
                "captions": [item["caption"] for item in batch],
                "references": [item["references"] for item in batch],
                "file_names": [item["file_name"] for item in batch],
                "image_ids": [item["image_id"] for item in batch],
            }

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    results = []
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            predictions = generate_captions(
                model, processor, tokenizer, batch, args, device
            )

            for pred, refs, fname, img_id in zip(
                predictions,
                batch["references"],
                batch["file_names"],
                batch["image_ids"],
            ):
                results.append(
                    {
                        "image_id": img_id,
                        "file_name": fname,
                        "prediction": pred.strip(),
                        "references": refs,
                    }
                )

    total_time = time.time() - start_time
    fps = len(results) / total_time

    print(f"Total time: {total_time:.2f}s")
    print(f"FPS: {fps:.2f} imgs/s")

    print("Computing metrics...")
    metrics = compute_metrics(results)

    if wandb_cfg["enabled"]:
        wandb.log(
            {
                "metrics/bleu1": metrics["bleu1"],
                "metrics/bleu2": metrics["bleu2"],
                "metrics/rougeL": metrics["rougeL"],
                "metrics/meteor": metrics["meteor"],
                "performance/fps": fps,
                "performance/total_time_s": total_time,
                "performance/num_images": len(results),
            }
        )
        wandb.finish()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Saved {len(results)} predictions → {output_path}")
