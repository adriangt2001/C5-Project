
import torch
from diffusers import (AutoPipelineForText2Image,
                       AutoPipelineForImage2Image,
                       DiffusionPipeline,
                       Flux2KleinPipeline,
                       Flux2Pipeline,
                       FluxPipeline,
                       AutoModel)

from transformers import Mistral3ForConditionalGeneration


def run_diffusion_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model_name = args.model_name

    if model_name == "stabilityai/sd-turbo":

        if args.image_prompt:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_name, torch_dtype=torch.float16, variant="fp16")
            pipe.to("cuda")

            # TODO: add image input
            init_image = None

            image = pipe(args.prompt, image=init_image, num_inference_steps=2,
                         strength=0.5, guidance_scale=0.0).images[0]

        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_name, torch_dtype=torch.float16, variant="fp16")
            pipe.to("cuda")

            image = pipe(prompt=args.prompt,
                         num_inference_steps=args.num_inference_steps,
                         guidance_scale=0.0).images[0]

    elif model_name == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to("cuda")

        # if using torch < 2.0
        # pipe.enable_xformers_memory_efficient_attention()

        images = pipe(prompt=args.prompt).images[0]

    elif model_name == "black-forest-labs/FLUX.1-schnell":

        pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=dtype)
        # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        pipe.enable_model_cpu_offload()

        image = pipe(
            prompt=args.prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

    elif model_name == "diffusers/FLUX.2-dev-bnb-4bit":

        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=dtype, device_map="cpu"
        )
        dit = AutoModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=dtype, device_map="cpu"
        )
        pipe = Flux2Pipeline.from_pretrained(
            model_name, text_encoder=text_encoder, transformer=dit, torch_dtype=dtype
        )
        pipe.enable_model_cpu_offload()

        # TODO: add image input
        input_image = None

        image = pipe(
            prompt=args.prompt,
            # image=[input_image] #multi-image input
            generator=torch.Generator(device=device).manual_seed(42),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=4,
        ).images[0]

        image.save("flux2_output.png")

    elif model_name == "black-forest-labs/FLUX.2-klein-base-4B":
        pipe = Flux2KleinPipeline.from_pretrained(
            model_name, torch_dtype=dtype)
        pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

        image = pipe(
            prompt=args.prompt,
            height=1024,
            width=1024,
            guidance_scale=4.0,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(0)
        ).images[0]

    else:
        raise ValueError(f"Unsupported model: {model_name}")
