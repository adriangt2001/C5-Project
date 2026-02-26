import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.models import FasterRCNN
from src.custom_datasets import KittiDatasetTorchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Perspective(p=0.1),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.log_wandb:
        wandb.init(
            project="C5-Week1", 
            entity="c5-team2", 
            name=f"Finetune-{args.model}-{args.variant}-first-try",
            config=args
        )

    train_ds = KittiDatasetTorchvision(
        root=args.dataset, 
        annotations_folder=args.annotation_folder, 
        image_folder=args.image_folder,
        seqmap_file="src/custom_datasets/train.seqmap", 
        transform=get_train_transforms()
    )
    val_ds = KittiDatasetTorchvision(
        root=args.dataset, 
        annotations_folder=args.annotation_folder, 
        image_folder=args.image_folder,
        seqmap_file="src/custom_datasets/val.seqmap", 
        transform=get_val_transforms()
    )

    collate_fn = lambda x: tuple(zip(*x))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3 classes: car, pedestrian + background
    detector = FasterRCNN(variant=args.variant, num_classes=3, device=device)
    model = detector.model
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_map = 0
    for epoch in range(int(args.epochs)):
        detector.set_train_mode()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        lr_scheduler.step()

        # Evaluation
        detector.set_eval_mode()
        metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
        
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Eval]"):
            detector.evaluate(images, targets, metric)
        
        results = metric.compute()

        print(f"\n{'='*30}")
        print(f" Epoch {epoch} Summary ")
        print(f"{'='*30}")
        print(f" Train Loss:    {epoch_loss / len(train_loader):.4f}")
        print(f" mAP @.50:.95:  {results['map']:.4f}")
        print(f" mAP @.50:      {results['map_50']:.4f}")
        print(f" mAP Car:       {results['map_per_class'][0]:.4f}")
        print(f" mAP Pedestrian: {results['map_per_class'][1]:.4f}")
        
        metrics_log = {
            "train/loss": epoch_loss / len(train_loader),
            "mAP/main": results["map"],
            "mAP/50": results["map_50"],
            "mAP/class_Car": results["map_per_class"][0],
            "mAP/class_Pedestrian": results["map_per_class"][1],
            "epoch": epoch
        }

        if args.log_wandb:
            wandb.log(metrics_log)

        if results["map"] > best_map:
            best_map = results["map"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/fasterrcnn_{args.variant}_best.pth")

    if args.log_wandb:
        wandb.finish()
