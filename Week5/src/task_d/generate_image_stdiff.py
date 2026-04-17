from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import ensure_dir, load_config, save_json
from src.utils.pipeline import (
    get_torch_dtype,
    load_generation_pipeline,
    resolve_device,
    seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTIONS_PATH = PROJECT_ROOT / "src" / "task_d" / "captions.json"
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_PROMPT_PREFIX = "low quality picture of "
DEFAULT_PROMPT_SUFFIX = ""

MODEL_OPTIONS = {
    "sd15": {
        "label": "Stable Diffusion v1.5",
        "model_id": "runwayml/stable-diffusion-v1-5",
        "family": "sd",
        "height": 512,
        "width": 512,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "dtype": "float16",
    },
    "sd21": {
        "label": "Stable Diffusion 2.1",
        "model_id": "Manojb/stable-diffusion-2-1-base",
        "family": "sd",
        "height": 768,
        "width": 768,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "dtype": "float16",
    },
    "sdxl": {
        "label": "SDXL Base 1.0",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "family": "sdxl",
        "height": 1024,
        "width": 1024,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "dtype": "float16",
    },
    "sdxl_turbo": {
        "label": "SDXL Turbo",
        "model_id": "stabilityai/sdxl-turbo",
        "family": "sdxl",
        "height": 1024,
        "width": 1024,
        "guidance_scale": 0.0,
        "num_inference_steps": 4,
        "dtype": "float16",
    },
    "sd3_medium": {
        "label": "Stable Diffusion 3 Medium",
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "family": "sd3",
        "height": 1024,
        "width": 1024,
        "guidance_scale": 7.0,
        "num_inference_steps": 28,
        "dtype": "float16",
    },
}


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def sanitize_name(value: str, max_length: int = 80) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return sanitized[:max_length] or "item"


def resolve_path(path_value: str | None, default: Path | None = None) -> Path | None:
    if path_value is None:
        return default
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_caption_records(captions_path: Path) -> list[dict]:
    with open(captions_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = []

    if isinstance(payload, dict) and isinstance(payload.get("captions_by_theme"), dict):
        for theme, items in payload["captions_by_theme"].items():
            if not isinstance(items, list):
                continue
            for idx, item in enumerate(items):
                if isinstance(item, dict):
                    caption = item.get("caption", "")
                    if isinstance(caption, str) and caption.strip():
                        records.append(
                            {
                                "caption": caption.strip(),
                                "theme": theme,
                                "source_file_name": item.get("file_name"),
                                "source_index": idx,
                            }
                        )
                elif isinstance(item, str) and item.strip():
                    records.append(
                        {
                            "caption": item.strip(),
                            "theme": theme,
                            "source_file_name": None,
                            "source_index": idx,
                        }
                    )
    elif isinstance(payload, dict) and isinstance(payload.get("captions"), list):
        for idx, caption in enumerate(payload["captions"]):
            if isinstance(caption, str) and caption.strip():
                records.append(
                    {
                        "caption": caption.strip(),
                        "theme": None,
                        "source_file_name": None,
                        "source_index": idx,
                    }
                )
    elif isinstance(payload, list):
        for idx, caption in enumerate(payload):
            if isinstance(caption, str) and caption.strip():
                records.append(
                    {
                        "caption": caption.strip(),
                        "theme": None,
                        "source_file_name": None,
                        "source_index": idx,
                    }
                )
    else:
        raise ValueError(f"Unsupported captions structure in {captions_path}")

    if not records:
        raise ValueError(f"No captions found in {captions_path}")

    return records


def build_prompt(caption: str, prompt_prefix: str, prompt_suffix: str) -> str:
    return f"{prompt_prefix}{caption}{prompt_suffix}".strip()


def build_runtime_config(config: dict) -> dict:
    model_option = config.get("model_option", "sdxl_turbo")
    preset = MODEL_OPTIONS.get(model_option)
    if preset is None:
        available = ", ".join(sorted(MODEL_OPTIONS))
        raise ValueError(
            f"Unknown model_option '{model_option}'. Available options: {available}"
        )

    model_id = config.get("model_id", preset["model_id"])
    family = config.get("family", preset["family"])
    dtype_name = config.get("dtype", preset["dtype"])
    model_name = config.get("model_name") or sanitize_name(model_option)

    output_dir = resolve_path(config.get("output_dir"))
    if output_dir is None:
        output_dir = DEFAULT_OUTPUTS_ROOT / model_name

    runtime = {
        "model_option": model_option,
        "model_label": preset["label"],
        "model_name": model_name,
        "model_id": model_id,
        "family": family,
        "dtype": dtype_name,
        "device": config.get("device", "auto"),
        "captions_path": resolve_path(config.get("captions_path"), DEFAULT_CAPTIONS_PATH),
        "output_dir": output_dir,
        "images_dir": output_dir / "images",
        "manifest_path": resolve_path(config.get("manifest_path"), output_dir / "captions_with_images.json"),
        "report_path": resolve_path(config.get("report_path"), output_dir / "generation_report.jsonl"),
        "prompt_prefix": config.get("prompt_prefix", DEFAULT_PROMPT_PREFIX),
        "prompt_suffix": config.get("prompt_suffix", DEFAULT_PROMPT_SUFFIX),
        "negative_prompt": config.get("negative_prompt"),
        "seed": int(config.get("seed", 42)),
        "limit": config.get("limit"),
        "skip_existing": bool(config.get("skip_existing", False)),
        "height": int(config.get("height", preset["height"])),
        "width": int(config.get("width", preset["width"])),
        "guidance_scale": float(config.get("guidance_scale", preset["guidance_scale"])),
        "num_inference_steps": int(
            config.get("num_inference_steps", preset["num_inference_steps"])
        ),
    }
    return runtime


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_generate_image_stdiff(config_path: str) -> None:
    config = load_config(config_path)
    runtime = build_runtime_config(config)

    ensure_dir(runtime["images_dir"])
    caption_records = load_caption_records(runtime["captions_path"])
    if runtime["limit"] is not None:
        caption_records = caption_records[: int(runtime["limit"])]

    device = resolve_device(runtime["device"])
    torch_dtype = get_torch_dtype(runtime["dtype"])

    log(f"Loading model option '{runtime['model_option']}' -> {runtime['model_id']}")
    log(f"Captions source: {runtime['captions_path']}")
    log(f"Output directory: {runtime['output_dir']}")

    pipeline = load_generation_pipeline(
        model_id=runtime["model_id"],
        family=runtime["family"],
        device=device,
        torch_dtype=torch_dtype,
    )

    run_started = datetime.now(timezone.utc).isoformat()
    append_jsonl(
        runtime["report_path"],
        {
            "event": "run_started",
            "timestamp_utc": run_started,
            "config_path": str(config_path),
            "model_option": runtime["model_option"],
            "model_id": runtime["model_id"],
            "family": runtime["family"],
            "device": device,
            "dtype": runtime["dtype"],
            "num_prompts": len(caption_records),
            "output_dir": str(runtime["output_dir"]),
        },
    )

    manifest_records = []

    for idx, record in enumerate(caption_records):
        caption = record["caption"]
        prompt = build_prompt(
            caption,
            runtime["prompt_prefix"],
            runtime["prompt_suffix"],
        )
        seed = runtime["seed"] + idx
        prompt_slug = sanitize_name(caption, max_length=50)
        image_path = runtime["images_dir"] / f"{idx:05d}_{prompt_slug}.png"

        entry = {
            "index": idx,
            "caption": caption,
            "prompt": prompt,
            "theme": record["theme"],
            "source_file_name": record["source_file_name"],
            "source_index": record["source_index"],
            "seed": seed,
            "image_path": str(image_path),
            "status": None,
            "generation_time_seconds": None,
            "error": None,
        }

        if runtime["skip_existing"] and image_path.exists():
            entry["status"] = "skipped_existing"
            manifest_records.append(entry)
            append_jsonl(
                runtime["report_path"],
                {
                    "event": "skipped_existing",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    **entry,
                },
            )
            log(f"Skipping existing image {image_path.name}")
            continue

        generator = seed_everything(seed, device)
        params = {
            "prompt": prompt,
            "negative_prompt": runtime["negative_prompt"],
            "height": runtime["height"],
            "width": runtime["width"],
            "guidance_scale": runtime["guidance_scale"],
            "num_inference_steps": runtime["num_inference_steps"],
            "generator": generator,
        }

        start = time.perf_counter()
        status = "success"
        error_message = None

        try:
            sync_device(device)
            result = pipeline(**params)
            sync_device(device)
            result.images[0].save(image_path)
        except Exception as exc:
            status = "error"
            error_message = str(exc)

        elapsed = round(time.perf_counter() - start, 4)
        entry["status"] = status
        entry["generation_time_seconds"] = elapsed
        entry["error"] = error_message
        manifest_records.append(entry)

        append_jsonl(
            runtime["report_path"],
            {
                "event": "image_finished",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **entry,
            },
        )

        if status == "success":
            log(f"[{idx + 1}/{len(caption_records)}] saved {image_path.name} in {elapsed:.2f}s")
        else:
            log(f"[{idx + 1}/{len(caption_records)}] failed in {elapsed:.2f}s: {error_message}")

    summary = {
        "task": "task_d_stable_diffusion_generation",
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "captions_path": str(runtime["captions_path"]),
        "model_option": runtime["model_option"],
        "model_label": runtime["model_label"],
        "model_name": runtime["model_name"],
        "model_id": runtime["model_id"],
        "family": runtime["family"],
        "device": device,
        "dtype": runtime["dtype"],
        "prompt_prefix": runtime["prompt_prefix"],
        "prompt_suffix": runtime["prompt_suffix"],
        "negative_prompt": runtime["negative_prompt"],
        "seed": runtime["seed"],
        "height": runtime["height"],
        "width": runtime["width"],
        "guidance_scale": runtime["guidance_scale"],
        "num_inference_steps": runtime["num_inference_steps"],
        "output_dir": str(runtime["output_dir"]),
        "images_dir": str(runtime["images_dir"]),
        "report_path": str(runtime["report_path"]),
        "num_captions": len(caption_records),
        "num_successes": sum(item["status"] == "success" for item in manifest_records),
        "num_errors": sum(item["status"] == "error" for item in manifest_records),
        "num_skipped": sum(item["status"] == "skipped_existing" for item in manifest_records),
        "generations": manifest_records,
    }
    save_json(runtime["manifest_path"], summary)
    append_jsonl(
        runtime["report_path"],
        {
            "event": "run_finished",
            "timestamp_utc": summary["run_finished_utc"],
            "num_successes": summary["num_successes"],
            "num_errors": summary["num_errors"],
            "num_skipped": summary["num_skipped"],
            "manifest_path": str(runtime["manifest_path"]),
        },
    )
    log(f"Saved manifest to {runtime['manifest_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Task D images with a configurable Stable Diffusion model."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()
    run_generate_image_stdiff(args.config)


if __name__ == "__main__":
    main()
