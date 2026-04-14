from __future__ import annotations

from pathlib import Path
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
from PIL import Image
from transformers import AutoModel, Mistral3ForConditionalGeneration


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "diffusion_generation"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_SEED = 0


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
    for model_name in model_names:
        image = _generate_with_model(model_name, args, device)
        model_slug = _sanitize_filename(model_name, max_length=60)
        output_path = output_dir / f"{model_slug}__seed_{args.seed}__{prompt_slug}.png"
        image.save(output_path)
        saved_paths.append(output_path)
        print(f"Saved generation for {model_name} -> {output_path}")

    return saved_paths
