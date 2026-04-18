from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.task_e.finetuining_sd_data import (
    PROJECT_ROOT,
    VizWizCaptionDatasetWithSyntheticDirs,
    normalize_finetuning_args,
)
from src.utils import compute_metrics, load_config
from src.utils.dataset import collate_fn, load_annotations, split_train_val


THEMES = {
    "quality_unclear": [
        "quality issues",
        "blurry",
        "too blurry",
        "cannot tell",
        "hard to see",
        "too dark",
        "too bright",
        "unrecognizable",
        "not clear",
    ],
    "screens_devices": [
        "screen",
        "computer",
        "monitor",
        "laptop",
        "windows",
        "dialog",
        "tv",
        "television",
        "keyboard",
        "phone",
        "captcha",
    ],
    "documents_text": [
        "receipt",
        "paper",
        "label",
        "text",
        "card",
        "book",
        "coupon",
        "instructions",
        "poster",
        "sign",
        "barcode",
        "mail",
        "letter",
        "package",
        "manual",
    ],
    "food_drink": [
        "food",
        "bottle",
        "drink",
        "soda",
        "can",
        "medicine",
        "cereal",
        "snack",
        "meat",
        "plate",
        "cup",
        "popcorn",
    ],
    "clothing_body": [
        "shirt",
        "sweater",
        "shoe",
        "sock",
        "foot",
        "feet",
        "pants",
        "bra",
        "fabric",
        "clothing",
        "dress",
        "hand",
        "arm",
    ],
    "household_objects": [
        "vacuum",
        "dryer",
        "dishwasher",
        "toilet",
        "filter",
        "chair",
        "table",
        "desk",
        "counter",
        "appliance",
        "machine",
        "box",
    ],
    "outdoor_transport": [
        "bus",
        "car",
        "truck",
        "building",
        "tree",
        "street",
        "restaurant",
        "mall",
        "outdoor",
        "mailbox",
    ],
}

THEME_PRIORITY = [
    "quality_unclear",
    "screens_devices",
    "documents_text",
    "food_drink",
    "clothing_body",
    "household_objects",
    "outdoor_transport",
    "other",
]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def assign_theme_disjoint(references: list[str]) -> str:
    reference_text = " ".join(references).lower()
    scores = {
        theme: sum(keyword in reference_text for keyword in keywords)
        for theme, keywords in THEMES.items()
    }
    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return "other"

    tied_themes = [theme for theme in THEME_PRIORITY if scores.get(theme, 0) == best_score]
    return tied_themes[0]


def build_val_loader(args, processor) -> DataLoader:
    data_dir = resolve_path(args.data_dir)
    val_annotations_path = getattr(args, "val_annotations_path", None)
    if val_annotations_path:
        val_samples = load_annotations(resolve_path(val_annotations_path))
    else:
        samples = load_annotations(data_dir / "annotations" / "train_filtered.json")
        _, val_samples = split_train_val(
            samples,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )

    max_len = min(processor.tokenizer.model_max_length, 40)
    val_dataset = VizWizCaptionDatasetWithSyntheticDirs(
        data_dir=data_dir,
        samples=val_samples,
        processor=processor,
        max_len=max_len,
        training=False,
    )
    return DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )


def evaluate_checkpoint(model, processor, val_loader, max_new_tokens: int, num_beams: int):
    device = next(model.parameters()).device
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
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

    return results


def compute_theme_table(results: list[dict]) -> pd.DataFrame:
    classified_results = []
    for item in results:
        classified_results.append(
            {
                **item,
                "theme": assign_theme_disjoint(item["references"]),
            }
        )

    theme_metrics = []
    for theme in THEME_PRIORITY:
        subset = [row for row in classified_results if row["theme"] == theme]
        if not subset:
            continue

        metrics = compute_metrics(subset)
        theme_metrics.append(
            {
                "theme": theme,
                "count": len(subset),
                "bleu1": round(float(metrics["bleu1"]), 4),
                "bleu2": round(float(metrics["bleu2"]), 4),
                "rougeL": round(float(metrics["rougeL"]), 4),
                "meteor": round(float(metrics["meteor"]), 4),
            }
        )

    return pd.DataFrame(theme_metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a finetuned BLIP checkpoint and print per-theme captioning metrics."
    )
    parser.add_argument(
        "--config",
        default="configs/task_e/blip-finetuning-sd-data.yaml",
        help="Path to the training config used for the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="finetuned_models/sd_data",
        help="Directory containing config.json, processor files, and model.safetensors.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/task_c/sd_data_theme_eval.json",
        help="Where to save the full evaluation payload.",
    )
    parser.add_argument(
        "--val-annotations",
        default="data/annotations/val.json",
        help="Annotation file to use for evaluation. Defaults to the official validation split.",
    )
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    eval_args = argparse.Namespace(**config)
    eval_args = normalize_finetuning_args(eval_args)
    eval_args.val_annotations_path = args.val_annotations

    checkpoint_dir = resolve_path(args.checkpoint)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(checkpoint_dir)
    model = BlipForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        use_safetensors=True,
    )
    model.to(device)

    val_loader = build_val_loader(eval_args, processor)
    results = evaluate_checkpoint(
        model=model,
        processor=processor,
        val_loader=val_loader,
        max_new_tokens=eval_args.max_new_tokens,
        num_beams=eval_args.num_beams,
    )
    overall_metrics = compute_metrics(results)
    theme_metrics_df = compute_theme_table(results)

    print("Overall validation metrics:")
    for key, value in overall_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nPer-theme captioning metrics:")
    print(theme_metrics_df.to_string(index=True))

    payload = {
        "checkpoint": str(checkpoint_dir),
        "config": str(resolve_path(args.config)),
        "overall_metrics": {
            key: round(float(value), 4) for key, value in overall_metrics.items()
        },
        "theme_metrics": theme_metrics_df.to_dict(orient="records"),
        "results": results,
    }
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved evaluation payload to: {output_json}")


if __name__ == "__main__":
    main()
