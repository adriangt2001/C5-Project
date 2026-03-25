
def load_model_and_processor(model_type: str, model_name: str, device: str, mode: str = "inference"):
    if model_type:
        if model_type == "vit-gpt2":
            from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
            model = VisionEncoderDecoderModel.from_pretrained(model_name, use_safetensors=True)
            processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        elif model_type == "blip":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            tokenizer = None  # blip uses the processor for everything

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
