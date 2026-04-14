from __future__ import annotations

from pathlib import Path
import math
import re
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


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "diffusion_generation"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_SEED = 0
GRID_CELL_SIZE = 512
GRID_PADDING = 24
GRID_HEADER_HEIGHT = 140
GRID_LABEL_HEIGHT = 72


def _sanitize_filename(value: str, max_length: int = 80) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return sanitized[:max_length] or "generation"


def _resolve_models(args):
    if getattr(args, "model_names", None):
        return list(args.model_names)
    if getattr(args, "model_name", None):
        return [args.model_name]
    raise ValueError("A diffusion model must be provided via 'model_name' or 'model_names'.")


def _resolve_output_dir(args) -> Path:
    output_dir = Path(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_image_prompt(image_prompt: Optional[str]) -> Optional[Image.Image]:
    if not image_prompt:
        return None
    return Image.open(image_prompt).convert("RGB")


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


def _build_comparison_grid(saved_images, prompt: str, seed: int, output_dir: Path, prompt_slug: str):
    if not saved_images:
        return None

    font = ImageFont.load_default()
    cols = min(3, len(saved_images))
    rows = math.ceil(len(saved_images) / cols)
    cell_width = GRID_CELL_SIZE
    cell_height = GRID_CELL_SIZE + GRID_LABEL_HEIGHT
    canvas_width = GRID_PADDING * (cols + 1) + cell_width * cols
    canvas_height = GRID_HEADER_HEIGHT + GRID_PADDING * (rows + 1) + cell_height * rows

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(245, 244, 239))
    draw = ImageDraw.Draw(canvas)

    header_x = GRID_PADDING
    header_width = canvas_width - 2 * GRID_PADDING
    prompt_lines = _wrap_text(f"Prompt: {prompt}", draw, font, header_width)
    draw.text((header_x, 20), f"Diffusion Model Comparison | seed={seed}", fill=(30, 30, 30), font=font)

    current_y = 48
    for line in prompt_lines:
        draw.text((header_x, current_y), line, fill=(60, 60, 60), font=font)
        current_y += 16

    for index, item in enumerate(saved_images):
        row = index // cols
        col = index % cols
        origin_x = GRID_PADDING + col * (cell_width + GRID_PADDING)
        origin_y = GRID_HEADER_HEIGHT + GRID_PADDING + row * (cell_height + GRID_PADDING)

        image = item["image"].copy().convert("RGB")
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

        model_lines = _wrap_text(item["model_name"], draw, font, cell_width - 12)
        label_y = origin_y + GRID_CELL_SIZE + 10
        for line in model_lines[:3]:
            draw.text((origin_x + 6, label_y), line, fill=(40, 40, 40), font=font)
            label_y += 16

    grid_path = output_dir / f"comparison__seed_{seed}__{prompt_slug}.png"
    canvas.save(grid_path)
    print(f"Saved comparison grid -> {grid_path}")
    return grid_path


def _generate_with_model(model_name: str, args, device: torch.device):
    is_cuda = device.type == "cuda"
    pipe_dtype = torch.bfloat16 if is_cuda else torch.float32
    fp16_dtype = torch.float16 if is_cuda else torch.float32
    seed = getattr(args, "seed", DEFAULT_SEED)
    num_inference_steps = getattr(args, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
    generator = _generator_for(device, seed)

    if model_name == "stabilityai/sd-turbo":
        if args.image_prompt:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_name,
                **_pipeline_kwargs(fp16_dtype, is_cuda),
            )
            pipe = pipe.to(device)
            init_image = _load_image_prompt(args.image_prompt)
            image = pipe(
                prompt=args.prompt,
                image=init_image,
                num_inference_steps=min(num_inference_steps, 2),
                strength=0.5,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_name,
                **_pipeline_kwargs(fp16_dtype, is_cuda),
            )
            pipe = pipe.to(device)
            image = pipe(
                prompt=args.prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

    elif model_name == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            **_pipeline_kwargs(fp16_dtype, is_cuda, use_safetensors=True),
        )
        pipe = pipe.to(device)
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

    elif model_name == "black-forest-labs/FLUX.1-schnell":
        pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=pipe_dtype)
        if is_cuda:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        image = pipe(
            prompt=args.prompt,
            guidance_scale=0.0,
            num_inference_steps=min(num_inference_steps, 4),
            max_sequence_length=256,
            generator=generator,
        ).images[0]

    elif model_name == "diffusers/FLUX.2-dev-bnb-4bit":
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

        image = pipe(
            prompt=args.prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=4,
        ).images[0]

    elif model_name == "black-forest-labs/FLUX.2-klein-base-4B":
        pipe = Flux2KleinPipeline.from_pretrained(model_name, torch_dtype=pipe_dtype)
        if is_cuda:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        image = pipe(
            prompt=args.prompt,
            height=1024,
            width=1024,
            guidance_scale=4.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return image


def run_diffusion_inference(args):
    if not getattr(args, "prompt", None):
        raise ValueError("A non-empty prompt must be provided.")

    if getattr(args, "num_inference_steps", None) is None:
        args.num_inference_steps = DEFAULT_NUM_INFERENCE_STEPS
    if getattr(args, "seed", None) is None:
        args.seed = DEFAULT_SEED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = _resolve_output_dir(args)
    model_names = _resolve_models(args)
    prompt_slug = _sanitize_filename(args.prompt, max_length=50)

    saved_paths = []
    saved_images = []
    for model_name in model_names:
        image = _generate_with_model(model_name, args, device)
        model_slug = _sanitize_filename(model_name, max_length=60)
        output_path = output_dir / f"{model_slug}__seed_{args.seed}__{prompt_slug}.png"
        image.save(output_path)
        saved_paths.append(output_path)
        saved_images.append(
            {
                "model_name": model_name,
                "image": image,
                "path": output_path,
            }
        )
        print(f"Saved generation for {model_name} -> {output_path}")

    comparison_path = _build_comparison_grid(
        saved_images,
        prompt=args.prompt,
        seed=args.seed,
        output_dir=output_dir,
        prompt_slug=prompt_slug,
    )
    if comparison_path is not None:
        saved_paths.append(comparison_path)

    return saved_paths
