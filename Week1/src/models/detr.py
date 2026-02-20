from huggingface_hub import interpreter_login
from .register import register_model, BaseModel
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from transformers.pipelines.pt_utils import KeyDataset
import torch
from PIL import Image


class DeTR(BaseModel):
    def __init__(self, device=None):
        self.device = device
        interpreter_login()

        self.image_processor = AutoImageProcessor.from_pretrained('microsoft/conditional-detr-resnet-50')
        self.model = AutoModelForObjectDetection.from_pretrained('microsoft/conditional-detr-resnet-50')
        self.model.to(self.device)

    def train(self):
        pass

    def __label2id(self, label: str):
        return 0

    def __list2dict(self, l: list):
        d = {
            'scores': [],
            'labels': [],
            'boxes': []
        }

        for element in l:
            d['scores'].append(element['score'])
            d['labels'].append(self._label2id(element['label']))
            d['boxes'].append([
                element['box']['xmin'],
                element['box']['ymin'],
                element['box']['xmax'],
                element['box']['ymax'],
            ])

        d['scores'] = torch.as_tensor(d['scores'])
        d['labels'] = torch.as_tensor(d['labels'])
        d['boxes'] = torch.as_tensor(d['boxes'])

        return d

    def inference(self, images: list, reset=False):
        inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)

        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        sizes = torch.as_tensor([img.shape[1:] for img in images])
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=0.0)

        return results