import torch
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

def build_fasterrcnn(variant: str):
    """
    Returns: (model, weights)
    """
    v = variant.lower()

    if v == "resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        return model, weights

    if v == "resnet50_fpn_v2":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        return model, weights

    if v == "mobilenet_v3_large_fpn":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        return model, weights

    if v == "mobilenet_v3_large_320_fpn":
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        return model, weights
    
    raise ValueError(f"Unknown variant '{variant}'")

class FasterRCNN:
    """
    Torchvision Faster R-CNN detector (COCO pretrained).
    """
    def __init__(self, variant="resnet50_fpn_v2", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, weights = build_fasterrcnn(variant)
        self.model = model.to(self.device).eval()
        self.categories = weights.meta["categories"] 
        self.preprocess = weights.transforms()
        self._to_tensor = ToTensor() 
        # Task d: Mapping COCo_KITTI-MOTS
        self.kitti_mapping = {1: 2, 3: 1}

    @torch.inference_mode()
    def inference(self, images):
        """
        Performs inference on a batch of images.
        Args:
            images: List of PIL images or Tensors.
        Returns:
            List of dictionaries containing 'boxes', 'labels', and 'scores'.
        """
        # If single image, convert to list
        if not isinstance(images, (list, tuple)):
            images = [images]

        processed_images = []
        for img in images:
            if not isinstance(img, torch.Tensor):
                img = self._to_tensor(img)
            processed_images.append(self.preprocess(img).to(self.device))

        predictions = self.model(processed_images)
        return predictions

    def evaluate(self, images, targets, metric):
        """
        Processes a batch for evaluation, applying the class mapping and 
        updating the provided torchmetrics MeanAveragePrecision object.
        
        Args:
            images: Batch of images from DataLoader.
            targets: Batch of ground truth dictionaries from DataLoader.
            metric: torchmetrics.detection.mean_ap.MeanAveragePrecision instance.
        """
        preds = self.inference(images)

        processed_preds = []
        for p in preds:
            # Keep only classes present in our mapping (Pedestrian and Car)
            keep = [i for i, label in enumerate(p['labels']) if label.item() in self.kitti_mapping]
            
            processed_preds.append({
                'boxes': p['boxes'][keep].cpu(),
                'scores': p['scores'][keep].cpu(),
                'labels': torch.tensor([self.kitti_mapping[l.item()] for l in p['labels'][keep]], dtype=torch.int64)
            })

        targets_cpu = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        metric.update(processed_preds, targets_cpu)