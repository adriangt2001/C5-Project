from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import AutoModel, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration


DEFAULT_MODEL_ID = "diffusers/FLUX.2-dev-bnb-4bit"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve(
).parents[2] / "results" / "flux2_dev_generation"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sanitize_filename(value: str, max_length: int = 80) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return sanitized[:max_length] or "prompt"


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_prompts(args) -> list[dict]:
    prompt_records = []
    rng = random.Random(args.seed)

    if args.prompts_json:
        prompts_json_path = resolve_path(args.prompts_json)
        log(f"Loading prompts from JSON: {prompts_json_path}")
        if prompts_json_path is None or not prompts_json_path.exists():
            raise FileNotFoundError(
                f"Prompts JSON file not found: {args.prompts_json} "
                f"(resolved to {prompts_json_path})"
            )
        payload = read_json(prompts_json_path)

        if isinstance(payload, list):
            for idx, item in enumerate(payload):
                if isinstance(item, str) and item.strip():
                    prompt_records.append(
                        {
                            "prompt": item.strip(),
                            "theme": None,
                            "source_file_name": None,
                            "source_index": idx,
                        }
                    )
        elif isinstance(payload, dict):
            captions_by_theme = payload.get("captions_by_theme")

            if isinstance(captions_by_theme, dict):
                themed_records = {}
                for theme, items in captions_by_theme.items():
                    if not isinstance(items, list):
                        continue

                    theme_records = []
                    for idx, item in enumerate(items):
                        if isinstance(item, str) and item.strip():
                            theme_records.append(
                                {
                                    "prompt": item.strip(),
                                    "theme": theme,
                                    "source_file_name": None,
                                    "source_index": idx,
                                }
                            )
                        elif isinstance(item, dict):
                            caption = item.get("caption", "")
                            if isinstance(caption, str) and caption.strip():
                                theme_records.append(
                                    {
                                        "prompt": caption.strip(),
                                        "theme": theme,
                                        "source_file_name": item.get("file_name"),
                                        "source_index": idx,
                                    }
                                )

                    if theme_records:
                        rng.shuffle(theme_records)
                        themed_records[theme] = theme_records

                # Round-robin over themes so early interruption still leaves a balanced set.
                theme_names = sorted(themed_records.keys())
                while True:
                    added_any = False
                    for theme in theme_names:
                        if themed_records[theme]:
                            prompt_records.append(themed_records[theme].pop())
                            added_any = True
                    if not added_any:
                        break

            elif isinstance(payload.get("captions"), list):
                for idx, item in enumerate(payload["captions"]):
                    if isinstance(item, str) and item.strip():
                        prompt_records.append(
                            {
                                "prompt": item.strip(),
                                "theme": None,
                                "source_file_name": None,
                                "source_index": idx,
                            }
                        )

        else:
            raise ValueError(
                f"Unsupported prompts JSON structure in {args.prompts_json}")

    if args.prompts:
        log(f"Loading {len(args.prompts)} prompts from --prompts")
        for idx, prompt in enumerate(args.prompts):
            if prompt.strip():
                prompt_records.append(
                    {
                        "prompt": prompt.strip(),
                        "theme": None,
                        "source_file_name": None,
                        "source_index": idx,
                    }
                )

    if not prompt_records:
        raise ValueError(
            "No prompts were loaded. "
            f"prompts_json={getattr(args, 'prompts_json', None)!r}, "
            f"num_prompts_cli={len(getattr(args, 'prompts', []) or [])}. "
            "Provide prompts via --prompts or --prompts_json."
        )

    if args.limit is not None:
        prompt_records = prompt_records[: args.limit]
        log(f"Applied prompt limit: {args.limit}")

    log(f"Loaded {len(prompt_records)} prompts")

    return prompt_records


def build_model_prompt(caption: str, prompt_prefix: str | None) -> str:
    prefix = (prompt_prefix or "").strip()
    if not prefix:
        return caption
    return f"{prefix}{caption}"


def load_pipeline(model_id: str, device: torch.device, torch_dtype: torch.dtype) -> Flux2Pipeline:
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    transformer = AutoModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    pipe = Flux2Pipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )

    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_task_d_flux(args) -> None:
    output_dir = resolve_path(args.output_dir) or DEFAULT_OUTPUT_DIR
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    report_path = resolve_path(args.report_path) if args.report_path else output_dir / "generation_report.jsonl"
    log("Starting FLUX image generation task")
    log(f"Resolved output directory: {output_dir}")
    log(f"Resolved report path: {report_path}")
    log(
        "Generation config: "
        f"model_id={args.model_id}, device={args.device}, seed={args.seed}, "
        f"steps={args.num_inference_steps}, guidance_scale={args.guidance_scale}, "
        f"height={args.height}, width={args.width}, prompt_prefix={getattr(args, 'prompt_prefix', '')!r}"
    )
    prompt_records = load_prompts(args)

    device = resolve_device(args.device)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    log(f"Resolved device: {device}")
    log(f"Loading model {args.model_id} on {device}")
    pipeline = load_pipeline(args.model_id, device, torch_dtype)
    log(f"Loaded model {args.model_id}")

    run_metadata = {
        "event": "run_started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "device": str(device),
        "num_prompts": len(prompt_records),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "prompt_prefix": getattr(args, "prompt_prefix", ""),
        "height": args.height,
        "width": args.width,
        "output_dir": str(output_dir),
    }
    append_jsonl(report_path, run_metadata)
    log(f"Wrote run start metadata to {report_path}")

    for idx, record in enumerate(prompt_records):
        caption = record["prompt"]
        model_prompt = build_model_prompt(caption, getattr(args, "prompt_prefix", ""))
        prompt_slug = sanitize_filename(caption, max_length=50)
        image_path = images_dir / f"{idx:05d}_{prompt_slug}.png"
        image_seed = args.seed + idx
        log(
            f"Starting prompt {idx + 1}/{len(prompt_records)} | "
            f"theme={record['theme']} | file={record['source_file_name']} | seed={image_seed}"
        )
        append_jsonl(
            report_path,
            {
                "event": "generation_started",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "index": idx,
                "caption": caption,
                "model_prompt": model_prompt,
                "theme": record["theme"],
                "source_file_name": record["source_file_name"],
                "seed": image_seed,
                "target_image_path": str(image_path),
            },
        )

        if args.skip_existing and image_path.exists():
            append_jsonl(
                report_path,
                {
                    "event": "skipped_existing",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "index": idx,
                    "caption": caption,
                    "model_prompt": model_prompt,
                    "theme": record["theme"],
                    "source_file_name": record["source_file_name"],
                    "seed": image_seed,
                    "image_path": str(image_path),
                },
            )
            log(f"Skipping existing image: {image_path}")
            continue

        start = time.perf_counter()
        status = "success"
        error_message = None

        try:
            generator = make_generator(device, image_seed)
            sync_device(device)
            result = pipeline(
                prompt=model_prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )
            sync_device(device)
            image = result.images[0]
            image.save(image_path)
        except Exception as exc:
            status = "error"
            error_message = str(exc)

        elapsed = time.perf_counter() - start

        report_record = {
            "event": "image_finished",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "index": idx,
            "caption": caption,
            "model_prompt": model_prompt,
            "theme": record["theme"],
            "source_file_name": record["source_file_name"],
            "seed": image_seed,
            "generation_time_seconds": round(elapsed, 4),
            "image_path": str(image_path) if status == "success" else None,
            "error": error_message,
        }
        append_jsonl(report_path, report_record)

        if status == "success":
            log(f"[{idx + 1}/{len(prompt_records)}] saved {image_path.name} in {elapsed:.2f}s")
        else:
            log(f"[{idx + 1}/{len(prompt_records)}] failed in {elapsed:.2f}s: {error_message}")

    append_jsonl(
        report_path,
        {
            "event": "run_finished",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": args.model_id,
            "num_prompts": len(prompt_records),
        },
    )
    log(f"Finished. Images are in {images_dir}")
    log(f"Report written incrementally to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images with diffusers/FLUX.2-dev-bnb-4bit.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--prompts_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--prompt_prefix", type=str, default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    run_task_d_flux(parser.parse_args())
