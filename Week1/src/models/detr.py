import torch
from huggingface_hub import interpreter_login
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class DeTR:
    def __init__(self, variant='facebook/detr-resnet-50', threshold=0.5, device=None):
        self.device = device
        self.threshold = threshold
        interpreter_login()

        self.image_processor = AutoImageProcessor.from_pretrained(variant)
        self.model = AutoModelForObjectDetection.from_pretrained(variant)
        self.model.to(self.device)

    def train(self):
        pass

    def get_labels(self):
        return self.model.config.id2label

    def inference(self, images: list):
        inputs = self.image_processor(images=images, return_tensors="pt")

        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        sizes = torch.as_tensor([img.shape[1:] for img in images])
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=self.threshold)

        return results

if __name__ == '__main__':
    model = DeTR()
    labels = model.get_labels()
    print(labels)