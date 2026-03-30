import torch


def load_model_and_processor(
    model_type: str, model_name: str, device: str, mode: str = "inference"
):
    if model_type:
        if model_type == "vit-gpt2":
            from transformers import (
                VisionEncoderDecoderModel,
                AutoImageProcessor,
                AutoTokenizer,
            )

            model = VisionEncoderDecoderModel.from_pretrained(
                model_name, use_safetensors=True
            )
            processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        elif model_type == "blip":
            from transformers import BlipProcessor, BlipForConditionalGeneration

            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(
                model_name, use_safetensors=True
            )
            tokenizer = None  # blip uses the processor for everything

        elif model_type == "qwen3.5_9b":
            from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

            processor = AutoProcessor.from_pretrained(model_name)
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            tokenizer = None

        elif model_type == "vit-qwen4":
            pass

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.to(device)

        if mode == "inference":
            model.eval()

        elif mode == "finetuning":
            model.train()

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return model, processor, tokenizer
