import time
from pathlib import Path

from PIL import Image

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from src.utils import (
    ensure_dir,
    get_torch_dtype,
    load_config,
    make_image_grid,
    resolve_device,
    save_json,
    seed_everything,
)


def generate_compare_inputs(config_path: str):
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)
    device = resolve_device(config["device"])

    model_cfg = config["model"]
    # Load both pipelines
    text2img_pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=get_torch_dtype(config["dtype"]),
        use_safetensors=True,
    ).to(device)

    img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=get_torch_dtype(config["dtype"]),
        use_safetensors=True,
    ).to(device)

    # Disable progress bars
    text2img_pipeline.set_progress_bar_config(disable=True)
    img2img_pipeline.set_progress_bar_config(disable=True)

    init_image = None
    if config.get("image_path"):
        image_path = Path(config["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        init_image = Image.open(image_path).convert("RGB")

    results = []

    for mode in config.get("modes", ["caption_only"]):
        mode_dir = output_dir / mode
        ensure_dir(mode_dir)

        for prompt_idx, caption in enumerate(config.get("captions", [])):
            # Generate multiple images with different seeds for variety
            image_paths = []
            generation_times = []

            for image_idx in range(config["num_images_per_prompt"]):
                # Use different seed for each image: base_seed + prompt_idx * 100 + image_idx
                image_seed = config["seed"] + prompt_idx * 100 + image_idx
                generator = seed_everything(image_seed, device)

                if mode == "caption_only":
                    pipeline = text2img_pipeline
                    params = {
                        "prompt": caption,
                        "negative_prompt": config.get("negative_prompt", None),
                        "height": model_cfg["height"],
                        "width": model_cfg["width"],
                        "guidance_scale": model_cfg["guidance_scale"],
                        "num_inference_steps": model_cfg["num_inference_steps"],
                        "num_images_per_prompt": 1,  # Generate one at a time for different seeds
                        "generator": generator,
                    }
                elif mode == "caption_and_image":
                    pipeline = img2img_pipeline
                    if init_image is None:
                        raise ValueError("caption_and_image mode requires image_path in config")
                    # Vary strength slightly for more diversity
                    strength_variation = 0.6 + (image_idx * 0.1)  # 0.6, 0.7, 0.8, 0.9, 1.0, etc.
                    strength_variation = min(strength_variation, 1.0)  # Cap at 1.0

                    params = {
                        "prompt": caption,
                        "image": init_image,
                        "negative_prompt": config.get("negative_prompt", None),
                        "strength": strength_variation,
                        "guidance_scale": model_cfg["guidance_scale"],
                        "num_inference_steps": model_cfg["num_inference_steps"],
                        "num_images_per_prompt": 1,  # Generate one at a time
                        "generator": generator,
                    }
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                start = time.perf_counter()
                result = pipeline(**params)
                elapsed = time.perf_counter() - start
                generation_times.append(elapsed)

                # Save individual image
                image_name = f"prompt_{prompt_idx:02d}_image_{image_idx:02d}.png"
                image_path = mode_dir / image_name
                result.images[0].save(image_path)
                image_paths.append(str(image_path))

            # Create grid with all images
            if image_paths:
                images = [Image.open(path) for path in image_paths]
                title = f"Mode: {mode} | Prompt: {caption[:50]}..."
                grid = make_image_grid(images, title=title)
                grid.save(mode_dir / f"prompt_{prompt_idx:02d}_grid.png")

                # Calculate average generation time
                avg_time = sum(generation_times) / len(generation_times)

            results.append(
                {
                    "mode": mode,
                    "prompt_index": prompt_idx,
                    "prompt": caption,
                    "image_path": config.get("image_path"),
                    "guidance_scale": model_cfg["guidance_scale"],
                    "num_inference_steps": model_cfg["num_inference_steps"],
                    "generation_time_seconds": elapsed,
                    "image_paths": image_paths,
                }
            )

    summary = {
        "task": "task_a_compare_inputs",
        "model": model_cfg["name"],
        "model_id": model_cfg["model_id"],
        "device": device,
        "results": results,
    }
    save_json(output_dir / "summary.json", summary)
    print(f"Saved compare_inputs summary in {output_dir / 'summary.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare caption only vs caption+image inputs for SDXL")
    parser.add_argument("--config", required=True, help="Path to compare_inputs.yaml")
    args = parser.parse_args()

    generate_compare_inputs(args.config)
