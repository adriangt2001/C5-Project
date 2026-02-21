from huggingface_hub import interpreter_login
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from transformers.pipelines.pt_utils import KeyDataset
import torch
from PIL import Image


class DeTR:
    def __init__(self, device=None):
        self.device = device
        interpreter_login()

        self.image_processor = AutoImageProcessor.from_pretrained('microsoft/conditional-detr-resnet-50')
        self.model = AutoModelForObjectDetection.from_pretrained('microsoft/conditional-detr-resnet-50')
        self.model.to(self.device)

    def train(self):
        pass

    def get_labels(self):
        return self.model.config.id2label

    def inference(self, images: list, reset=False):
        inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)

        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        sizes = torch.as_tensor([img.shape[1:] for img in images])
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=0.0)

        return results

if __name__ == '__main__':
    model = DeTR()
    labels = model.get_labels()
    print(labels)