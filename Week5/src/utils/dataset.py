from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

# Adapted from Week3/src/dataset.py
# Main changes:
#   - Removed SimpleTokenizer (HuggingFace tokenizer used instead)
#   - Replaced manual transforms with AutoImageProcessor (we can use another)
#   - __getitem__ no longer encodes captions (done by model internally)
#   - collate_fn removed input_ids and target_ids


@dataclass
class VizWizSample:
    image_id: int
    file_name: str
    split: str
    captions: List[str]
    text_detected: bool


# Load annotations
def load_annotations(annotation_path: Path) -> List[VizWizSample]:
    payload = json.loads(annotation_path.read_text())
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    # split = annotation_path.stem
    # TODO: change flux
    split = "train" if "train_filtered" == annotation_path.stem \
            else "synthetic" if "train_synthetic" == annotation_path.stem \
            else "synthetic_sd" if "train_synthetic_sd" == annotation_path.stem \
            else annotation_path.stem

    captions_by_image: Dict[int, List[str]] = defaultdict(list)
    for annotation in annotations:
        if annotation.get("is_rejected"):
            continue
        caption = annotation.get("caption", "").strip()
        if caption:
            captions_by_image[annotation["image_id"]].append(caption)

    samples: List[VizWizSample] = []
    for image in images:
        samples.append(
            VizWizSample(
                image_id=image["id"],
                file_name=image["file_name"],
                split=split,
                captions=captions_by_image.get(image["id"], []),
                text_detected=bool(image.get("text_detected", False)),
            )
        )
    return samples


# train/val split --> 10% of training for validation
def split_train_val(
    samples: Sequence[VizWizSample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[VizWizSample], List[VizWizSample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_at = int(len(indices) * (1.0 - val_ratio))

    train_samples = [samples[idx] for idx in indices[:split_at]]
    val_samples = [samples[idx] for idx in indices[split_at:]]
    return train_samples, val_samples


# Our Dataset
class VizWizCaptionDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        samples: Sequence[VizWizSample],
        processor: AutoImageProcessor,
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

    def _resolve_image_path(self, sample: VizWizSample) -> Path:
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
        samples: Sequence[VizWizSample],
    ) -> List[VizWizSample]:
        kept_samples: List[VizWizSample] = []
        missing_samples: List[VizWizSample] = []

        for sample in samples:
            image_path = self._resolve_image_path(sample)
            if image_path.exists():
                kept_samples.append(sample)
            else:
                missing_samples.append(sample)

        if missing_samples:
            split_counts: Dict[str, int] = defaultdict(int)
            for sample in missing_samples:
                split_counts[sample.split] += 1
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
            caption = (
                random.choice(
                    sample.captions) if self.training else sample.captions[0]
            )

        if self.is_llm:
            return {
                "image": image,
                "caption": caption,
                "references": list(sample.captions),
                "file_name": sample.file_name,
                "image_id": sample.image_id,
            }
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0)

            return {
                "pixel_values": pixel_values,
                "caption": caption,
                "references": list(sample.captions),
                "file_name": sample.file_name,
                "image_id": sample.image_id,
            }


def collate_fn(batch: Sequence[Dict]) -> Dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "captions": [item["caption"] for item in batch],
        "references": [item["references"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
    }


def train_collate_fn(
    batch: Sequence[Dict],
    processor: AutoImageProcessor,
    max_len: int = 40,
) -> Dict:
    captions = [item["caption"] for item in batch]
    text_inputs = processor.tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    labels = text_inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "captions": captions,
        "references": [item["references"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "labels": labels,
    }


def build_train_collate_fn(
    processor: AutoImageProcessor,
    max_len: int = 40,
):
    return partial(train_collate_fn, processor=processor, max_len=max_len)
