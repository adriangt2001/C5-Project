from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import wandb

from src.utils.models import load_model_and_processor
from src.utils.dataset import (
    VizWizSample,
    load_annotations,
    split_train_val,
    build_train_collate_fn,
    collate_fn,
)
from src.utils import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log_with_time(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_finetuning_args(args):
    """Coerce config values to numeric types expected by PyTorch."""

    args.batch_size = int(args.batch_size)
    args.num_workers = int(args.num_workers)
    args.split_seed = int(args.split_seed)
    args.max_new_tokens = int(args.max_new_tokens)
    args.num_beams = int(args.num_beams)
    args.epochs = int(args.epochs)
    args.val_ratio = float(args.val_ratio)
    args.lr_encoder = float(args.lr_encoder)
    args.lr_decoder = float(args.lr_decoder)
    args.weight_decay = float(args.weight_decay)
    args.use_synthetic = bool(getattr(args, "use_synthetic", False))
    synthetic_data_dirs = getattr(args, "synthetic_data_dirs", None)
    if isinstance(synthetic_data_dirs, str):
        synthetic_data_dirs = [synthetic_data_dirs]
    args.synthetic_data_dirs = synthetic_data_dirs or []
    return args


@dataclass
class SyntheticVizWizSample:
    image_id: int
    file_name: str
    split: str
    captions: List[str]
    text_detected: bool
    image_path: Path


class VizWizCaptionDatasetWithSyntheticDirs(Dataset):
    def __init__(
        self,
        data_dir: Path,
        samples: Sequence[VizWizSample | SyntheticVizWizSample],
        processor,
        max_len: int = 40,
        is_llm: bool = False,
        training: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_len = max_len
        self.is_llm = is_llm
        self.training = training
        self.samples = self._filter_missing_samples(list(samples))

    def _resolve_image_path(
        self, sample: VizWizSample | SyntheticVizWizSample
    ) -> Path:
        synthetic_path = getattr(sample, "image_path", None)
        if synthetic_path is not None:
            return Path(synthetic_path)

        candidates = [
            self.data_dir / sample.split / sample.file_name,
            self.data_dir / "train" / sample.file_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _filter_missing_samples(
        self,
        samples: Sequence[VizWizSample | SyntheticVizWizSample],
    ) -> List[VizWizSample | SyntheticVizWizSample]:
        kept_samples = []
        missing_samples = []

        for sample in samples:
            image_path = self._resolve_image_path(sample)
            if image_path.exists():
                kept_samples.append(sample)
            else:
                missing_samples.append(sample)

        if missing_samples:
            split_counts: Dict[str, int] = {}
            for sample in missing_samples:
                split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
            split_summary = ", ".join(
                f"{split}={count}" for split, count in sorted(split_counts.items())
            )
            print(
                f"[dataset] Skipping {len(missing_samples)} samples with missing image files "
                f"({split_summary}).",
                flush=True,
            )

        return kept_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image_path = self._resolve_image_path(sample)
        image = Image.open(image_path).convert("RGB")
        caption = ""
        if sample.captions:
            caption = random.choice(sample.captions) if self.training else sample.captions[0]

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "references": list(sample.captions),
            "file_name": sample.file_name,
            "image_id": sample.image_id,
        }


def load_synthetic_samples_from_dir(
    synthetic_dir: Path,
    start_image_id: int,
) -> List[SyntheticVizWizSample]:
    manifest_path = synthetic_dir / "captions_with_images.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing captions_with_images.json in synthetic directory: {synthetic_dir}"
        )

    payload = json.loads(manifest_path.read_text())
    generations = payload.get("generations", [])

    samples: List[SyntheticVizWizSample] = []
    next_image_id = start_image_id
    synthetic_split = f"synthetic_{synthetic_dir.name}"

    for generation in generations:
        if generation.get("status") != "success":
            continue

        image_path_value = generation.get("image_path")
        caption = (generation.get("caption") or "").strip()
        if not image_path_value or not caption:
            continue

        image_path = Path(image_path_value)
        if not image_path.is_absolute():
            image_path = synthetic_dir / image_path
        samples.append(
            SyntheticVizWizSample(
                image_id=next_image_id,
                file_name=image_path.name,
                split=synthetic_split,
                captions=[caption],
                text_detected=False,
                image_path=image_path,
            )
        )
        next_image_id += 1

    return samples


def build_dataloaders(args, processor):
    """Build train and validation dataloaders for finetuning."""

    data_dir = resolve_path(args.data_dir)
    samples = load_annotations(data_dir / "annotations" / "train_filtered.json")
    train_samples, val_samples = split_train_val(
        samples,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )

    log_with_time(
        f"Loaded original dataset from {data_dir / 'annotations' / 'train_filtered.json'}"
    )
    log_with_time(f"Original train samples: {len(train_samples)}")
    log_with_time(f"Validation samples: {len(val_samples)}")

    synthetic_samples: List[SyntheticVizWizSample] = []
    if args.use_synthetic and args.synthetic_data_dirs:
        next_image_id = max(sample.image_id for sample in train_samples + val_samples) + 1
        for synthetic_dir_value in args.synthetic_data_dirs:
            synthetic_dir = resolve_path(synthetic_dir_value)
            dir_samples = load_synthetic_samples_from_dir(synthetic_dir, next_image_id)
            synthetic_samples.extend(dir_samples)
            next_image_id += len(dir_samples)
            log_with_time(
                f"Loaded {len(dir_samples)} synthetic samples from "
                f"{synthetic_dir / 'captions_with_images.json'}"
            )

        train_samples += synthetic_samples
    elif args.use_synthetic:
        synthetic_samples = load_annotations(data_dir / "annotations" / "train_synthetic.json")
        train_samples += synthetic_samples
        log_with_time(
            f"Loaded synthetic dataset from {data_dir / 'annotations' / 'train_synthetic.json'}"
        )
        log_with_time(f"Synthetic training samples: {len(synthetic_samples)}")
    else:
        log_with_time("No synthetic data added. Training will use only original images.")

    log_with_time(
        f"Final training samples: {len(train_samples)} "
        f"(original={len(train_samples) - len(synthetic_samples)}, synthetic={len(synthetic_samples)})"
    )

    max_len = min(processor.tokenizer.model_max_length, 40)

    train_dataset = VizWizCaptionDatasetWithSyntheticDirs(
        data_dir=data_dir,
        samples=train_samples,
        processor=processor,
        max_len=max_len,
        training=True,
    )
    val_dataset = VizWizCaptionDatasetWithSyntheticDirs(
        data_dir=data_dir,
        samples=val_samples,
        processor=processor,
        max_len=max_len,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_train_collate_fn(
            processor=processor,
            max_len=train_dataset.max_len,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def set_finetuning(args, model):
    if args.model_type == "blip":
        vision_encoder = model.vision_model
        text_decoder = model.text_decoder
    elif args.model_type == "vit-gpt2":
        raise NotImplementedError("Finetuning not implemented for vit-gpt2 yet")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    if not args.finetune_encoder and not args.finetune_decoder:
        raise ValueError(
            "At least one of finetune_encoder or finetune_decoder must be True"
        )

    if args.finetune_encoder:
        log_with_time("Finetuning encoder...")
        for p in vision_encoder.parameters():
            p.requires_grad = True
    else:
        log_with_time("Freezing encoder...")
        for p in vision_encoder.parameters():
            p.requires_grad = False

    if args.finetune_decoder:
        log_with_time("Finetuning decoder...")
        for p in text_decoder.parameters():
            p.requires_grad = True
    else:
        log_with_time("Freezing decoder...")
        for p in text_decoder.parameters():
            p.requires_grad = False

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    encoder_trainable_params = sum(
        parameter.numel()
        for parameter in vision_encoder.parameters()
        if parameter.requires_grad
    )
    decoder_trainable_params = sum(
        parameter.numel()
        for parameter in text_decoder.parameters()
        if parameter.requires_grad
    )

    log_with_time(
        "model_params "
        f"total={total_params} "
        f"trainable={trainable_params} "
        f"encoder_trainable={encoder_trainable_params} "
        f"decoder_trainable={decoder_trainable_params}"
    )

    return vision_encoder, text_decoder


def train_step(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    log_with_time(f"Starting training epoch with {num_batches} batches...")

    for batch_idx, batch in enumerate(train_loader, start=1):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == num_batches:
            log_with_time(
                f"Training batch {batch_idx}/{num_batches} "
                f"(running_loss={total_loss / batch_idx:.4f})"
            )

    return total_loss / max(num_batches, 1)


def evaluate_step(model, processor, val_loader, args, device):
    model.eval()
    results = []
    num_batches = len(val_loader)

    log_with_time(f"Starting validation over {num_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(val_loader, desc="Validation", leave=False), start=1
        ):
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
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

            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == num_batches:
                log_with_time(
                    f"Validation batch {batch_idx}/{num_batches} "
                    f"(predictions_collected={len(results)})"
                )

    return compute_metrics(results)


def run_finetuning_sd_data(args):
    args = normalize_finetuning_args(args)

    wandb_cfg = args.wandb
    if wandb_cfg["enabled"]:
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=wandb_cfg["name"],
            config=vars(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        torch.cuda.init()

    model, processor, tokenizer = load_model_and_processor(
        args.model_type, args.model_name, device, mode="finetuning"
    )

    vision_encoder, text_decoder = set_finetuning(args, model)
    train_loader, val_loader = build_dataloaders(args, processor)
    log_with_time(
        f"Dataloaders ready: train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )

    param_groups = []
    if args.finetune_encoder:
        param_groups.append(
            {
                "params": [p for p in vision_encoder.parameters() if p.requires_grad],
                "lr": args.lr_encoder,
            }
        )
    if args.finetune_decoder:
        param_groups.append(
            {
                "params": [p for p in text_decoder.parameters() if p.requires_grad],
                "lr": args.lr_decoder,
            }
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    log_with_time("Computing baseline validation metrics before training...")
    baseline_metrics = evaluate_step(model, processor, val_loader, args, device)
    log_with_time("--- Baseline Validation Metrics ---")
    for k, v in baseline_metrics.items():
        log_with_time(f"{k}: {v:.4f}")
    if wandb_cfg["enabled"]:
        wandb.log({f"val/{k}": v for k, v in baseline_metrics.items()} | {"epoch": 0})
    best_score_meteor = baseline_metrics["meteor"]

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        log_with_time(f"Starting epoch {epoch}/{args.epochs}...")
        train_loss = train_step(train_loader, model, optimizer, device)
        log_with_time(f"Epoch {epoch} train_loss: {train_loss:.4f}")

        log_with_time(f"Computing validation metrics for epoch {epoch}...")
        metrics = evaluate_step(model, processor, val_loader, args, device)
        log_with_time(f"--- Validation Metrics For Epoch {epoch} ---")
        for k, v in metrics.items():
            log_with_time(f"{k}: {v:.4f}")

        if wandb_cfg["enabled"]:
            wandb.log({"train/loss": train_loss} | {f"val/{k}": v for k, v in metrics.items()})

        if metrics["meteor"] > best_score_meteor:
            checkpoint_dir = Path(args.best_trained_model_path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            log_with_time(
                f"New highest METEOR value achieved! Saving model "
                f"checkpoint in {checkpoint_dir}..."
            )
            best_score_meteor = metrics["meteor"]
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)

    if wandb_cfg["enabled"]:
        wandb.finish()
