import torch
from diffusers import (
    AutoPipelineForText2Image,
    DDIMScheduler,
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)


def resolve_device(device: str):
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return "cpu"
    return device


def get_torch_dtype(dtype_name: str):
    dtype_name = dtype_name.lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def seed_everything(seed: int, device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def load_generation_pipeline(model_id: str, family: str, device: str, torch_dtype):
    family = family.lower()
    device = resolve_device(device)

    if device == "cpu" and torch_dtype != torch.float32:
        print("Using float32 because the pipeline is running on CPU.")
        torch_dtype = torch.float32

    if "turbo" in model_id.lower():
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    elif family in {"sd3", "sd3.5"}:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    elif family == "sdxl":
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

    pipeline = pipeline.to(device)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)

    return pipeline


def get_scheduler(name: str, scheduler_config):
    name = name.lower()
    if name == "ddim":
        return DDIMScheduler.from_config(scheduler_config)
    if name == "ddpm":
        return DDPMScheduler.from_config(scheduler_config)
    if name in {"flow_match_euler", "flowmatch_euler", "fm_euler"}:
        return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    raise ValueError(f"Unsupported scheduler: {name}")
