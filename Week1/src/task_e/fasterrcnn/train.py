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
import time
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transforms():
    return A.Compose(
        [
            A.GaussianBlur(sigma_limit=[0.5, 1.0], p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Perspective(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
        ),
    )


def get_val_transforms():
    return A.Compose(
        [ToTensorV2()],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
        ),
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    set_seed(42)

    if args.log_wandb:
        run = wandb.init(
            project="C5-Week1",
            entity="c5-team2",
            config=vars(args),
        )
        wandb.run.name = f"{args.variant}_lr{args.lr}_unfreeze_depth{args.unfreeze_depth}"

        for k, v in dict(wandb.config).items():
            setattr(args, k, v)

        print("RUN CONFIG:", args.variant, args.lr, args.epochs, args.unfreeze_depth)

    train_ds = KittiDatasetTorchvision(
        root=args.dataset,
        annotations_folder=args.annotation_folder,
        image_folder=args.image_folder,
        seqmap_file="src/custom_datasets/train.seqmap",
        transform=get_train_transforms(),
    )

    val_ds = KittiDatasetTorchvision(
        root=args.dataset,
        annotations_folder=args.annotation_folder,
        image_folder=args.image_folder,
        seqmap_file="src/custom_datasets/val.seqmap",
        transform=get_val_transforms(),
    )

    collate_fn = lambda x: tuple(zip(*x))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3 classes: car, pedestrian + background
    detector = FasterRCNN(variant=args.variant, num_classes=3, device=device)
    model = detector.model

    # Freeze everything first
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Progressive unfreezing
    if args.unfreeze_depth >= 1:
        for p in model.backbone.body.layer4.parameters():
            p.requires_grad = True

    if args.unfreeze_depth >= 2:
        for p in model.backbone.body.layer3.parameters():
            p.requires_grad = True

    if args.unfreeze_depth >= 3:
        for p in model.backbone.body.layer2.parameters():
            p.requires_grad = True

    if args.unfreeze_depth >= 4:
        for p in model.backbone.parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    best_map = 0
    epochs_no_improve = 0
    patience = 1
    eval_freq = 5

    for epoch in range(int(args.epochs)):
        detector.set_train_mode()
        epoch_train_loss = 0.0

        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t_train0 = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_train_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        if use_cuda:
            torch.cuda.synchronize()
            t_train1 = time.perf_counter()

        epoch_val_loss = 0
        detector.set_train_mode()

        if use_cuda:
            torch.cuda.synchronize()
            t_val0 = time.perf_counter()

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict_val = model(images, targets)
                epoch_val_loss += sum(loss for loss in loss_dict_val.values()).item()

        if use_cuda:
            torch.cuda.synchronize()
            t_val1 = time.perf_counter()

        should_eval = ((epoch + 1) % eval_freq == 0 or epoch == int(args.epochs) - 1)

        results = None
        if should_eval:
            detector.set_eval_mode()
            metric = MeanAveragePrecision( box_format="xyxy", class_metrics=True)

            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Eval mAP]"):
                detector.evaluate(images, targets, metric)

            results = metric.compute()

            current_map = results["map"].item()
            if current_map > best_map:
                best_map = current_map
                epochs_no_improve = 0

                os.makedirs(f"checkpoints/{args.unfreeze_depth}/{args.lr}",exist_ok=True)
                torch.save( model.state_dict(), f"checkpoints/{args.unfreeze_depth}/{args.lr}/fasterrcnn_{args.variant}_best.pth")
                print(f"New Best mAP: {best_map:.4f}. Checkpoint saved.")
            else:
                epochs_no_improve += 1
                print(f"No improve. Patience: {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        lr_scheduler.step()

        print(f"\n{'='*30}")
        print(f" Epoch {epoch} Summary ")
        print(f"{'='*30}")
        print(f" Train Loss: {epoch_train_loss / len(train_loader):.4f}")
        print(f" Val Loss: {epoch_val_loss / len(val_loader):.4f}")

        metrics_to_log = {
            "train/epoch": epoch,
            "train/loss": epoch_train_loss / len(train_loader),
            "val/loss": epoch_val_loss / len(val_loader),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "performance/epoch_train_time": t_train1 - t_train0,
            "performance/epoch_val_time": t_val1 - t_val0,
        }

        if results is not None:
            print(f" mAP @.50:.95: {results['map']:.4f}")
            print(f" mAP @.50: {results['map_50']:.4f}")
            print(f" mAP Car: {results['map_per_class'][0]:.4f}")
            print(f" mAP Pedestrian: {results['map_per_class'][1]:.4f}")

            metrics_to_log.update({
                    "mAP/main": results["map"],
                    "mAP/50": results["map_50"],
                    "mAP/75": results["map_75"],
                    "mAP/class_Car": results["map_per_class"][0],
                    "mAP/class_Pedestrian": results["map_per_class"][1],
                    "mAP/small": results["map_small"],
                    "mAP/medium": results["map_medium"],
                    "mAP/large": results["map_large"],
                    "mAR/Det1": results["mar_1"],
                    "mAR/Det10": results["mar_10"],
                    "mAR/Det100": results["mar_100"],
                    "mAR/small": results["mar_small"],
                    "mAR/medium": results["mar_medium"],
                    "mAR/large": results["mar_large"],      
            })

        if args.log_wandb:
            wandb.log(metrics_to_log)
 
    os.makedirs(f"checkpoints/{args.unfreeze_depth}/{args.lr}", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{args.unfreeze_depth}/{args.lr}/fasterrcnn_{args.variant}_final_epoch.pth")

    if args.log_wandb:
        wandb.finish()