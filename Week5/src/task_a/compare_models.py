import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import (
    ensure_dir,
    get_torch_dtype,
    load_generation_pipeline,
    make_image_grid,
    resolve_device,
    save_json,
    seed_everything,
)


def estimate_pipeline_size_mb(pipeline):
    seen_params = set()
    total_bytes = 0

    for component in pipeline.components.values():
        if not hasattr(component, "parameters"):
            continue
        for param in component.parameters():
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)
            total_bytes += param.numel() * param.element_size()

    return total_bytes / (1024 ** 2)


def generate_for_model(model_cfg, args):
    model_name = model_cfg["name"]
    model_dir = Path(args.output_dir) / model_name
    ensure_dir(model_dir)

    load_start = time.perf_counter()
    pipeline = load_generation_pipeline(
        model_id=model_cfg["model_id"],
        family=model_cfg.get("family", "sd"),
        device=args.device,
        torch_dtype=get_torch_dtype(args.dtype),
    )
    load_time_seconds = time.perf_counter() - load_start
    estimated_model_size_mb = estimate_pipeline_size_mb(pipeline)

    model_guidance = model_cfg.get("guidance_scale", args.guidance_scale)
    model_steps = model_cfg.get("num_inference_steps", args.num_inference_steps)

    samples = []
    prompt_times = []
    for prompt_idx, prompt in enumerate(tqdm(args.prompts, desc=f"Task a - {model_name}")):
        generator = seed_everything(args.seed + prompt_idx, args.device)
        generation_start = time.perf_counter()
        result = pipeline(
            prompt=prompt,
            negative_prompt=getattr(args, "negative_prompt", None),
            height=args.height,
            width=args.width,
            guidance_scale=model_guidance,
            num_inference_steps=model_steps,
            num_images_per_prompt=args.num_images_per_prompt,
            generator=generator,
        )
        generation_time_seconds = time.perf_counter() - generation_start
        prompt_times.append(generation_time_seconds)

        image_paths = []
        for image_idx, image in enumerate(result.images):
            image_name = f"prompt_{prompt_idx:02d}_image_{image_idx:02d}.png"
            image_path = model_dir / image_name
            image.save(image_path)
            image_paths.append(str(image_path))

        if result.images:
            grid = make_image_grid(result.images)
            grid.save(model_dir / f"prompt_{prompt_idx:02d}_grid.png")

        samples.append(
            {
                "prompt_index": prompt_idx,
                "prompt": prompt,
                "guidance_scale": model_guidance,
                "num_inference_steps": model_steps,
                "generation_time_seconds": generation_time_seconds,
                "image_paths": image_paths,
            }
        )

    total_generation_time_seconds = sum(prompt_times)
    average_generation_time_seconds = total_generation_time_seconds / max(len(prompt_times), 1)

    return {
        "model_name": model_name,
        "model_id": model_cfg["model_id"],
        "family": model_cfg.get("family", "sd"),
        "load_time_seconds": load_time_seconds,
        "total_generation_time_seconds": total_generation_time_seconds,
        "average_generation_time_seconds": average_generation_time_seconds,
        "estimated_model_size_mb": estimated_model_size_mb,
        "samples": samples,
    }


def run_task_a(args):
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    args.device = resolve_device(args.device)

    summary = {
        "task": "task_a",
        "goal": "Compare open-source Stable Diffusion models for synthetic VizWiz-oriented data generation.",
        "selected_recommendation": "stabilityai/stable-diffusion-xl-base-1.0",
        "resolved_device": args.device,
        "experiments": [],
        "failed_models": [],
    }

    for model_cfg in args.models:
        try:
            summary["experiments"].append(generate_for_model(model_cfg, args))
        except Exception as exc:
            print(f"Skipping model {model_cfg['name']} because it failed: {exc}")
            summary["failed_models"].append(
                {
                    "model_name": model_cfg["name"],
                    "model_id": model_cfg["model_id"],
                    "error": str(exc),
                }
            )

    save_json(output_dir / "summary.json", summary)
    comparison_rows = []
    for experiment in summary["experiments"]:
        comparison_rows.append(
            {
                "model_name": experiment["model_name"],
                "model_id": experiment["model_id"],
                "family": experiment["family"],
                "resolved_device": args.device,
                "num_prompts": len(experiment["samples"]),
                "guidance_scale": experiment["samples"][0]["guidance_scale"] if experiment["samples"] else None,
                "num_inference_steps": experiment["samples"][0]["num_inference_steps"] if experiment["samples"] else None,
                "load_time_seconds": round(experiment["load_time_seconds"], 4),
                "total_generation_time_seconds": round(experiment["total_generation_time_seconds"], 4),
                "average_generation_time_seconds": round(experiment["average_generation_time_seconds"], 4),
                "estimated_model_size_mb": round(experiment["estimated_model_size_mb"], 2),
            }
        )

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df["avg_seconds_per_step"] = (
            comparison_df["average_generation_time_seconds"] / comparison_df["num_inference_steps"]
        ).round(4)
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    print(json.dumps(summary, indent=2))
