from itertools import product
from pathlib import Path

from tqdm import tqdm

from src.utils import (
    ensure_dir,
    get_scheduler,
    get_torch_dtype,
    load_generation_pipeline,
    resolve_device,
    save_json,
    seed_everything,
)


def run_task_b(args):
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    args.device = resolve_device(args.device)

    model_cfg = args.model
    pipeline = load_generation_pipeline(
        model_id=model_cfg["model_id"],
        family=model_cfg.get("family", "sdxl"),
        device=args.device,
        torch_dtype=get_torch_dtype(args.dtype),
    )

    experiments = []
    combinations = list(
        product(
            args.schedulers,
            args.guidance_scales,
            args.num_inference_steps,
            args.prompt_modes,
        )
    )

    for idx, (scheduler_name, guidance_scale, steps, prompt_mode) in enumerate(
        tqdm(combinations, desc="Task b")
    ):
        pipeline.scheduler = get_scheduler(scheduler_name, pipeline.scheduler.config)
        generator = seed_everything(args.seed + idx, args.device)

        negative_prompt = None
        if prompt_mode == "positive_and_negative":
            negative_prompt = args.negative_prompt

        experiment_name = (
            f"{scheduler_name}_cfg_{guidance_scale}_steps_{steps}_{prompt_mode}"
            .replace(".", "_")
        )
        experiment_dir = output_dir / experiment_name
        ensure_dir(experiment_dir)

        result = pipeline(
            prompt=args.base_prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )

        image_path = experiment_dir / "sample.png"
        result.images[0].save(image_path)

        experiments.append(
            {
                "experiment_name": experiment_name,
                "scheduler": scheduler_name,
                "guidance_scale": guidance_scale,
                "num_inference_steps": steps,
                "prompt_mode": prompt_mode,
                "prompt": args.base_prompt,
                "negative_prompt": negative_prompt,
                "image_path": str(image_path),
            }
        )

    summary = {
        "task": "task_b",
        "model_name": model_cfg["name"],
        "model_id": model_cfg["model_id"],
        "motivation": "Study how inference settings affect diffusion outputs before generating synthetic VizWiz-oriented samples.",
        "experiments": experiments,
    }
    save_json(output_dir / "summary.json", summary)
    print(f"Saved {len(experiments)} experiments to {output_dir}")
