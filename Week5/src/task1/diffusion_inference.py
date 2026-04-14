from __future__ import annotations

import json
from pathlib import Path
import math
import re
import time
from typing import Optional

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    Flux2KleinPipeline,
    Flux2Pipeline,
    FluxPipeline,
)
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, Mistral3ForConditionalGeneration

try:
    import wandb
except ImportError:
    wandb = None


DEFAULT_OUTPUT_DIR = Path(__file__).resolve(
).parents[2] / "results" / "diffusion_generation"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_SEED = 0
DEFAULT_IMAGE_STRENGTH = 0.5
GRID_CELL_SIZE = 512
GRID_PADDING = 24
GRID_HEADER_HEIGHT = 220
GRID_LABEL_HEIGHT = 150
GRID_MAX_SOURCE_SIZE = 768
GRID_TITLE_FONT_SIZE = 28
GRID_PROMPT_FONT_SIZE = 20
GRID_LABEL_FONT_SIZE = 18
GRID_TIME_FONT_SIZE = 18


def _sanitize_filename(value: str, max_length: int = 80) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return sanitized[:max_length] or "generation"


def _resolve_models(args):
    if getattr(args, "model_names", None):
        return list(args.model_names)
    if getattr(args, "model_name", None):
        return [args.model_name]
    raise ValueError(
        "A diffusion model must be provided via 'model_name' or 'model_names'.")


def _resolve_prompts(args):
    if getattr(args, "prompts", None):
        return [prompt for prompt in args.prompts if prompt]
    if getattr(args, "prompt", None):
        return [args.prompt]
    raise ValueError(
        "At least one non-empty prompt must be provided via 'prompt' or 'prompts'.")


def _resolve_output_dir(args, prompt: Optional[str] = None) -> Path:
    base_output_dir = Path(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR))
    base_output_dir.mkdir(parents=True, exist_ok=True)
    if not prompt:
        return base_output_dir

    prompt_slug = _sanitize_filename(prompt, max_length=50)
    prompt_output_dir = base_output_dir / prompt_slug
    prompt_output_dir.mkdir(parents=True, exist_ok=True)
    return prompt_output_dir


def _resolve_reference_images(args, prompts):
    reference_images = getattr(args, "reference_images", None)
    if not reference_images:
        return {prompt: None for prompt in prompts}
    if len(reference_images) != len(prompts):
        raise ValueError(
            f"'reference_images' must have the same length as 'prompts'. "
            f"Got {len(reference_images)} reference images for {len(prompts)} prompts."
        )
    return {
        prompt: reference_image if reference_image else None
        for prompt, reference_image in zip(prompts, reference_images)
    }


def _resolve_num_inference_steps(args, model_name: str) -> int:
    steps_by_model = getattr(args, "num_inference_steps_by_model", None) or {}
    if model_name in steps_by_model:
        return steps_by_model[model_name]
    return getattr(args, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)


def _load_image_prompt(image_prompt: Optional[str]) -> Optional[Image.Image]:
    if not image_prompt:
        return None
    return Image.open(image_prompt).convert("RGB")


def _load_reference_image(reference_image_path: Optional[str]) -> Optional[Image.Image]:
    if not reference_image_path:
        return None
    image_path = Path(reference_image_path)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Reference image not found: {reference_image_path}")
    return Image.open(image_path).convert("RGB")


def _ensure_supported_image_to_image_model(model_name: str):
    supported_models = {
        "stabilityai/sd-turbo",
        "stabilityai/sdxl-turbo",
        "stabilityai/stable-diffusion-xl-base-1.0",
    }
    if model_name not in supported_models:
        raise ValueError(
            f"Image-to-image is currently implemented only for {sorted(supported_models)}. "
            f"Model '{model_name}' is not supported in image-to-image mode yet."
        )


def _resize_reference_image(init_image: Image.Image, target_size: int = 1024) -> Image.Image:
    resized_image = init_image.copy()
    resized_image.thumbnail((target_size, target_size),
                            Image.Resampling.LANCZOS)
    width, height = resized_image.size
    width = max(8, (width // 8) * 8)
    height = max(8, (height // 8) * 8)
    if (width, height) != resized_image.size:
        resized_image = resized_image.resize(
            (width, height), Image.Resampling.LANCZOS)
    return resized_image


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _pipeline_kwargs(torch_dtype, is_cuda: bool, **extra_kwargs):
    kwargs = {"torch_dtype": torch_dtype, **extra_kwargs}
    if is_cuda:
        kwargs["variant"] = "fp16"
    return kwargs


def _wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current_line = words[0]
    for word in words[1:]:
        trial_line = f"{current_line} {word}"
        if draw.textbbox((0, 0), trial_line, font=font)[2] <= max_width:
            current_line = trial_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def _load_grid_font(size: int):
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, sample_text: str = "Ag") -> int:
    bbox = draw.textbbox((0, 0), sample_text, font=font)
    return bbox[3] - bbox[1]


def _build_comparison_grid(saved_images, prompt: str, seed: int, output_dir: Path, prompt_slug: str):
    if not saved_images:
        return None

    title_font = _load_grid_font(GRID_TITLE_FONT_SIZE)
    prompt_font = _load_grid_font(GRID_PROMPT_FONT_SIZE)
    label_font = _load_grid_font(GRID_LABEL_FONT_SIZE)
    time_font = _load_grid_font(GRID_TIME_FONT_SIZE)
    cols = min(3, len(saved_images))
    rows = math.ceil(len(saved_images) / cols)
    cell_width = GRID_CELL_SIZE
    cell_height = GRID_CELL_SIZE + GRID_LABEL_HEIGHT
    canvas_width = GRID_PADDING * (cols + 1) + cell_width * cols
    canvas_height = GRID_HEADER_HEIGHT + \
        GRID_PADDING * (rows + 1) + cell_height * rows

    canvas = Image.new("RGB", (canvas_width, canvas_height),
                       color=(245, 244, 239))
    draw = ImageDraw.Draw(canvas)

    header_x = GRID_PADDING
    header_width = canvas_width - 2 * GRID_PADDING
    prompt_lines = _wrap_text(
        f"Prompt: {prompt}", draw, prompt_font, header_width)
    draw.text((header_x, 20), f"Diffusion Model Comparison | seed={seed}", fill=(
        30, 30, 30), font=title_font)

    current_y = 20 + _line_height(draw, title_font) + 18
    for line in prompt_lines:
        draw.text((header_x, current_y), line,
                  fill=(60, 60, 60), font=prompt_font)
        current_y += _line_height(draw, prompt_font) + 8

    for index, item in enumerate(saved_images):
        row = index // cols
        col = index % cols
        origin_x = GRID_PADDING + col * (cell_width + GRID_PADDING)
        origin_y = GRID_HEADER_HEIGHT + GRID_PADDING + \
            row * (cell_height + GRID_PADDING)

        image = item["image"].copy().convert("RGB")
        image.thumbnail(
            (GRID_MAX_SOURCE_SIZE, GRID_MAX_SOURCE_SIZE), Image.Resampling.LANCZOS)
        image.thumbnail((cell_width, GRID_CELL_SIZE))
        image_x = origin_x + (cell_width - image.width) // 2
        image_y = origin_y + (GRID_CELL_SIZE - image.height) // 2

        draw.rectangle(
            [origin_x, origin_y, origin_x + cell_width, origin_y + GRID_CELL_SIZE],
            fill=(255, 255, 255),
            outline=(210, 210, 210),
            width=2,
        )
        canvas.paste(image, (image_x, image_y))

        label_text = item.get("display_name", item["model_name"])
        if item["model_name"] != "reference_image":
            label_text = f"model: {label_text}"
        model_lines = _wrap_text(
            label_text, draw, label_font, cell_width - 12)
        label_y = origin_y + GRID_CELL_SIZE + 10
        for line in model_lines[:3]:
            draw.text((origin_x + 6, label_y), line,
                      fill=(40, 40, 40), font=label_font)
            label_y += _line_height(draw, label_font) + 6
        if item.get("generation_time_s") is not None:
            draw.text(
                (origin_x + 6, label_y + 4),
                f"time: {item['generation_time_s']:.2f}s",
                fill=(70, 70, 70),
                font=time_font,
            )
            draw.text(
                (origin_x + 6, label_y + _line_height(draw, time_font) + 10),
                f"steps: {item['num_inference_steps']}",
                fill=(70, 70, 70),
                font=time_font,
            )

    grid_path = output_dir / f"comparison__seed_{seed}__{prompt_slug}.png"
    canvas.save(grid_path)
    print(f"Saved comparison grid -> {grid_path}")
    return grid_path


def _sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _get_wandb_config(args):
    return getattr(args, "wandb", {"enabled": False})


def _init_wandb(args):
    wandb_cfg = _get_wandb_config(args)
    if not wandb_cfg.get("enabled", False):
        return False
    if wandb is None:
        raise ImportError(
            "wandb logging is enabled in the config, but the 'wandb' package is not installed.")

    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=wandb_cfg["name"],
        config={
            "prompts": _resolve_prompts(args),
            "model_names": _resolve_models(args),
            "num_inference_steps": args.num_inference_steps,
            "num_inference_steps_by_model": getattr(args, "num_inference_steps_by_model", None),
            "seed": args.seed,
            "output_dir": str(Path(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR))),
            "image_prompt": args.image_prompt,
            "strength": getattr(args, "strength", DEFAULT_IMAGE_STRENGTH),
        },
    )
    return True


def _log_wandb_results(saved_images, comparison_path, args, prompt: str, prompt_slug: str):
    wandb_cfg = _get_wandb_config(args)
    if not wandb_cfg.get("enabled", False):
        return

    table = wandb.Table(
        columns=["prompt", "model_name", "generation_time_s", "image"])
    log_payload = {}
    for item in saved_images:
        generation_time_s = item.get("generation_time_s")
        display_name = item.get("display_name", item["model_name"])
        time_caption = "n/a" if generation_time_s is None else f"{generation_time_s:.2f}s"
        image_caption = (
            f"model={display_name} | prompt={prompt} | "
            f"seed={args.seed} | time={time_caption}"
        )
        wandb_image = wandb.Image(str(item["path"]), caption=image_caption)
        table.add_data(prompt, display_name, generation_time_s, wandb_image)

        model_slug = _sanitize_filename(item["model_name"], max_length=60)
        if generation_time_s is not None:
            log_payload[f"timing/{prompt_slug}/{model_slug}_seconds"] = generation_time_s
        log_payload[f"generation/{prompt_slug}/{model_slug}"] = wandb_image

    generated_images = [item for item in saved_images if item.get(
        "generation_time_s") is not None]
    if generated_images:
        total_generation_time_s = sum(
            item["generation_time_s"] for item in generated_images)
        avg_generation_time_s = total_generation_time_s / len(generated_images)
        log_payload[f"timing/{prompt_slug}/total_generation_time_s"] = total_generation_time_s
        log_payload[f"timing/{prompt_slug}/avg_generation_time_s"] = avg_generation_time_s
        log_payload[f"timing/{prompt_slug}/num_models"] = len(generated_images)
        log_payload[f"generations/{prompt_slug}/table"] = table

    if comparison_path is not None:
        log_payload[f"generation/{prompt_slug}/comparison_grid"] = wandb.Image(
            str(comparison_path),
            caption=f"prompt={prompt} | seed={args.seed}",
        )

    wandb.log(log_payload)


def _write_run_summary(output_dir: Path, prompt_results, args) -> Path:
    summary = {
        "seed": args.seed,
        "default_num_inference_steps": getattr(args, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS),
        "num_inference_steps_by_model": getattr(args, "num_inference_steps_by_model", None),
        "prompts": [],
    }

    for prompt, prompt_data in prompt_results.items():
        prompt_summary = {
            "prompt": prompt,
            "prompt_slug": prompt_data["prompt_slug"],
            "output_dir": str(prompt_data["output_dir"]),
            "reference_image": prompt_data["reference_image_path"],
            "generations": [],
        }

        for item in prompt_data["saved_images"]:
            if item["model_name"] == "reference_image":
                continue
            prompt_summary["generations"].append(
                {
                    "model_name": item["model_name"],
                    "generation_time_s": item["generation_time_s"],
                    "num_inference_steps": item["num_inference_steps"],
                    "image_path": str(item["path"]),
                }
            )

        summary["prompts"].append(prompt_summary)

    summary_path = output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved generation summary -> {summary_path}")
    return summary_path


def _load_model_bundle(model_name: str, args, device: torch.device):
    is_cuda = device.type == "cuda"
    pipe_dtype = torch.bfloat16 if is_cuda else torch.float32
    fp16_dtype = torch.float16 if is_cuda else torch.float32

    if model_name in ["stabilityai/sd-turbo", "stabilityai/sdxl-turbo"]:
        if args.image_prompt:
            _ensure_supported_image_to_image_model(model_name)
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_name,
                **_pipeline_kwargs(fp16_dtype, is_cuda),
            )
            pipe = pipe.to(device)
            bundle = {
                "model_name": model_name,
                "pipeline": pipe,
                "mode": "image_to_image_turbo",
                "init_image": _resize_reference_image(_load_image_prompt(args.image_prompt)),
            }
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_name,
                **_pipeline_kwargs(fp16_dtype, is_cuda),
            )
            pipe = pipe.to(device)
            bundle = {
                "model_name": model_name,
                "pipeline": pipe,
                "mode": "text_to_image",
            }

    elif model_name == "stabilityai/stable-diffusion-xl-base-1.0":
        if args.image_prompt:
            _ensure_supported_image_to_image_model(model_name)
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_name,
                **_pipeline_kwargs(fp16_dtype, is_cuda, use_safetensors=True),
            )
            pipe = pipe.to(device)
            bundle = {
                "model_name": model_name,
                "pipeline": pipe,
                "mode": "image_to_image_standard",
                "init_image": _resize_reference_image(_load_image_prompt(args.image_prompt)),
            }
        else:
            pipe = DiffusionPipeline.from_pretrained(
                model_name,
                **_pipeline_kwargs(fp16_dtype, is_cuda, use_safetensors=True),
            )
            pipe = pipe.to(device)
            bundle = {
                "model_name": model_name,
                "pipeline": pipe,
                "mode": "sdxl",
            }

    elif model_name == "black-forest-labs/FLUX.1-schnell":
        if args.image_prompt:
            _ensure_supported_image_to_image_model(model_name)
        pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=pipe_dtype)
        if is_cuda:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        bundle = {
            "model_name": model_name,
            "pipeline": pipe,
            "mode": "flux_schnell",
        }

    elif model_name == "diffusers/FLUX.2-dev-bnb-4bit":
        if args.image_prompt:
            _ensure_supported_image_to_image_model(model_name)
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=pipe_dtype,
            device_map="cpu",
        )
        dit = AutoModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=pipe_dtype,
            device_map="cpu",
        )
        pipe = Flux2Pipeline.from_pretrained(
            model_name,
            text_encoder=text_encoder,
            transformer=dit,
            torch_dtype=pipe_dtype,
        )
        if is_cuda:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        bundle = {
            "model_name": model_name,
            "pipeline": pipe,
            "mode": "flux2_dev",
            "text_encoder": text_encoder,
            "transformer": dit,
        }

    elif model_name == "black-forest-labs/FLUX.2-klein-base-4B":
        if args.image_prompt:
            _ensure_supported_image_to_image_model(model_name)
        pipe = Flux2KleinPipeline.from_pretrained(
            model_name, torch_dtype=pipe_dtype)
        if is_cuda:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        bundle = {
            "model_name": model_name,
            "pipeline": pipe,
            "mode": "flux2_klein",
        }

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return bundle


def _generate_with_model_bundle(model_bundle, prompt: str, args, device: torch.device):
    model_name = model_bundle["model_name"]
    pipe = model_bundle["pipeline"]
    seed = getattr(args, "seed", DEFAULT_SEED)
    num_inference_steps = _resolve_num_inference_steps(args, model_name)
    strength = getattr(args, "strength", DEFAULT_IMAGE_STRENGTH)
    generator = _generator_for(device, seed)

    print(
        f"Generating an image with {model_name} for prompt: {prompt}", flush=True)

    if model_bundle["mode"] == "image_to_image_turbo":
        image = pipe(
            prompt=prompt,
            image=model_bundle["init_image"],
            num_inference_steps=min(num_inference_steps, 2),
            strength=strength,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
    elif model_bundle["mode"] == "image_to_image_standard":
        image = pipe(
            prompt=prompt,
            image=model_bundle["init_image"],
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        ).images[0]
    elif model_bundle["mode"] == "text_to_image":
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
    elif model_bundle["mode"] == "sdxl":
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
    elif model_bundle["mode"] == "flux_schnell":
        image = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            num_inference_steps=min(num_inference_steps, 4),
            max_sequence_length=256,
            generator=generator,
        ).images[0]
    elif model_bundle["mode"] == "flux2_dev":
        image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=4,
        ).images[0]
    elif model_bundle["mode"] == "flux2_klein":
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=4.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
    else:
        raise ValueError(
            f"Unsupported generation mode: {model_bundle['mode']}")
    return image


def _release_model_bundle(model_bundle, device: torch.device):
    for key in ("pipeline", "text_encoder", "transformer", "init_image"):
        if key in model_bundle:
            del model_bundle[key]
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_diffusion_inference(args):
    if getattr(args, "num_inference_steps", None) is None:
        args.num_inference_steps = DEFAULT_NUM_INFERENCE_STEPS
    if getattr(args, "seed", None) is None:
        args.seed = DEFAULT_SEED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_output_dir = _resolve_output_dir(args)
    prompts = _resolve_prompts(args)
    reference_images_by_prompt = _resolve_reference_images(args, prompts)
    model_names = _resolve_models(args)
    wandb_enabled = _init_wandb(args)

    prompt_results = {}
    for prompt in prompts:
        prompt_slug = _sanitize_filename(prompt, max_length=50)
        prompt_results[prompt] = {
            "prompt_slug": prompt_slug,
            "output_dir": _resolve_output_dir(args, prompt),
            "saved_paths": [],
            "saved_images": [],
            "reference_image_path": reference_images_by_prompt[prompt],
        }

    all_saved_paths = []
    for model_name in model_names:
        print(f"Loading model once for all prompts: {model_name}", flush=True)
        model_bundle = _load_model_bundle(model_name, args, device)
        try:
            for prompt in prompts:
                prompt_data = prompt_results[prompt]
                _sync_device(device)
                start_time = time.perf_counter()
                image = _generate_with_model_bundle(
                    model_bundle, prompt, args, device)
                _sync_device(device)
                generation_time_s = time.perf_counter() - start_time

                model_slug = _sanitize_filename(model_name, max_length=60)
                output_path = prompt_data["output_dir"] / f"{model_slug}.png"
                image.save(output_path)
                all_saved_paths.append(output_path)
                prompt_data["saved_paths"].append(output_path)
                prompt_data["saved_images"].append(
                    {
                        "model_name": model_name,
                        "image": image,
                        "path": output_path,
                        "generation_time_s": generation_time_s,
                        "num_inference_steps": _resolve_num_inference_steps(args, model_name),
                    }
                )
                print(
                    f"Saved generation for {model_name} -> {output_path} ({generation_time_s:.2f}s)",
                    flush=True,
                )
        finally:
            _release_model_bundle(model_bundle, device)
            print(f"Released model resources for {model_name}", flush=True)

    for prompt in prompts:
        prompt_data = prompt_results[prompt]
        reference_image = _load_reference_image(
            prompt_data["reference_image_path"])
        if reference_image is not None:
            prompt_data["saved_images"].insert(
                0,
                {
                    "model_name": "reference_image",
                    "display_name": "example image of the VizWiz dataset (not used as input)",
                    "image": reference_image,
                    "path": prompt_data["reference_image_path"],
                    "generation_time_s": None,
                    "num_inference_steps": None,
                },
            )
        comparison_path = _build_comparison_grid(
            prompt_data["saved_images"],
            prompt=prompt,
            seed=args.seed,
            output_dir=prompt_data["output_dir"],
            prompt_slug=prompt_data["prompt_slug"],
        )
        if comparison_path is not None:
            all_saved_paths.append(comparison_path)
            prompt_data["saved_paths"].append(comparison_path)

        if wandb_enabled:
            _log_wandb_results(
                prompt_data["saved_images"],
                comparison_path,
                args,
                prompt,
                prompt_data["prompt_slug"],
            )

    if wandb_enabled:
        wandb.finish()

    summary_path = _write_run_summary(base_output_dir, prompt_results, args)
    all_saved_paths.append(summary_path)

    return all_saved_paths
