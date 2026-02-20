import torch
from torch.utils.data import DataLoader
from src.models.fasterrcnn import FasterRCNN
from src.custom_datasets.dataset_torchvision import KittiDatasetTorchvision
import argparse
import os
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

def main_inference(model_type="fasterrcnn", variant="resnet50_fpn", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "fasterrcnn":
        detector = FasterRCNN(variant=variant, device=device)
        ds = KittiDatasetTorchvision('dataset/KITTI-MOTS', 'instances_txt', 'training', 'src/custom_datasets/val.seqmap')
        coco_categories = detector.categories
        # Use zip(*) to handle variable numbers of objects per image and prevent tensor stacking errors
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x))) 
    
    elif model_type == "detr":
        pass

    elif model_type == "yolo":
        pass

    for i, (images, targets) in enumerate(tqdm(loader, desc="Running Inference")):
        preds = detector.inference(images)

        # Visualizing results for the first few batches
        if i < 3: 
            for j, (img, pred) in enumerate(zip(images, preds)):
                img_uint8 = (img * 255).to(torch.uint8)
                
                boxes = pred['boxes']
                labels = pred['labels']
                scores = pred['scores']
                
                label_names = [f"{coco_categories[l]}: {s:.2f}" for l, s in zip(labels, scores)]
                
                if len(boxes) > 0:
                    result_img = draw_bounding_boxes(
                        img_uint8, 
                        boxes=boxes, 
                        labels=label_names, 
                        colors="yellow", 
                        width=3
                    )
                    
                    os.makedirs("results/task_c/", exist_ok=True)
                    save_path = f"results/task_c/{model_type}_batch{i}_img{j}.png"
                    F.to_pil_image(result_img).save(save_path)

    print(f"Inference finished. Qualitative results saved in 'results/qualitative'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fastrrcnn", choices=["fasterrcnn", "detr", "yolo"])
    parser.add_argument("--variant", type=str, default="resnet50_fpn_v2")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    main_inference(model_type=args.model)