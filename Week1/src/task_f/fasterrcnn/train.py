import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.models import FasterRCNN
from src.custom_datasets import DEArtDatasetTorchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import time

from collections import Counter
import numpy as np
from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler based on object frequency.
    Images containing rare classes get higher probability.
    """

    print("Computing class frequencies...")

    class_counter = Counter()

    # Count object frequency per class
    for _, target in dataset:
        labels = target["labels"].tolist()
        class_counter.update(labels)

    print("Class distribution:", class_counter)

    # Compute inverse frequency weights
    total_objects = sum(class_counter.values())

    class_weights = {
        cls: total_objects / count
        for cls, count in class_counter.items()
    }

    # Compute weight per image
    sample_weights = []

    for _, target in dataset:
        labels = target["labels"].tolist()

        if len(labels) == 0:
            sample_weights.append(1.0)
        else:
            weights = [class_weights[l] for l in labels]
            sample_weights.append(np.mean(weights))

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler
    
def get_train_transforms_none():
    return A.Compose([
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True))

def get_train_transforms_mild():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True))

def get_train_transforms_aggressive():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(p=0.3),
        A.Perspective(p=0.5),
        A.Affine(scale=(0.8,1.2), rotate=(-15,15), p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True))

def get_val_transforms():
    return A.Compose([
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True))

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")

    # -------------------------------------------------
    # SELECT AUGMENTATION
    # -------------------------------------------------

    if args.aug == "none":
        train_transform = get_train_transforms_none()

    elif args.aug == "mild":
        train_transform = get_train_transforms_mild()

    elif args.aug == "aggressive":
        train_transform = get_train_transforms_aggressive()
    else:
        raise ValueError("Unknown augmentation type")

    # -------------------------------------------------
    # W&B INIT
    # -------------------------------------------------

    if args.log_wandb:
        run = wandb.init(
            project="C5-Week1",
            entity="c5-team2",
            config=vars(args),
        )
        wandb.run.name = f"{args.variant}_freeze{args.freeze_backbone}_lr{args.lr}_dataset{os.path.basename(args.dataset)}"
        for k, v in dict(wandb.config).items():
            setattr(args, k, v)

        print("RUN CONFIG:", args.variant, args.freeze_backbone, args.lr, args.epochs)

    # -------------------------------------------------
    # CHECKPOINT DIRECTORY STRUCTURE
    # -------------------------------------------------
    dataset_name = os.path.basename(args.dataset.rstrip("/"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"freeze{args.freeze_backbone}"
        f"_lr{args.lr}"
        f"_bs{args.batch_size}"
        f"_{timestamp}"
    )

    save_dir = os.path.join(
        "checkpoints",
        dataset_name,
        args.variant,
        run_name
    )

    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "best.pth")
    final_model_path = os.path.join(save_dir, "final.pth")

    # Save config for reproducibility
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    print(f"Models will be saved in: {save_dir}")
    # -------------------------------------------------
    # DATASETS (OUTSIDE wandb if!)
    # -------------------------------------------------
    train_ds = DEArtDatasetTorchvision(
        root=args.dataset,
        transform=train_transform,
        split="train"
    )

    val_ds = DEArtDatasetTorchvision(
        root=args.dataset,
        transform=get_val_transforms(),
        split="val"
    )

    collate_fn = lambda x: tuple(zip(*x))

    sampler = create_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,     
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    num_classes = len(train_ds.class_to_idx) + 1

    detector = FasterRCNN(
        variant=args.variant,
        num_classes=num_classes,
        device=device
    )

    model = detector.model
    model.roi_heads.positive_fraction = 0.5

    if args.freeze_backbone:
        print(f">>> Freezing backbone ({args.variant})")
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )

 
    best_map = 0
    patience = 1
    epochs_no_improve = 0
    eval_freq = 5

    # -------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------
    for epoch in range(int(args.epochs)):

        detector.set_train_mode()
        epoch_train_loss = 0.0

        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t_train0 = time.perf_counter()

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_train_loss += losses.item()

        if use_cuda:
            torch.cuda.synchronize()

        t_train1 = time.perf_counter()

        # -------------------------------------------------
        # VALIDATION LOSS
        # -------------------------------------------------
        detector.set_eval_mode()   # 🔥 FIXED
        epoch_val_loss = 0

        if use_cuda:
            torch.cuda.synchronize()

        t_val0 = time.perf_counter()

        # -------------------------------------------------
        # VALIDATION LOSS
        # -------------------------------------------------

        model.train()   
        epoch_val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):

                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict_val = model(images, targets)
                losses = sum(loss_dict_val.values())
                epoch_val_loss += losses.item()

        model.eval() 

        if use_cuda:
            torch.cuda.synchronize()

        t_val1 = time.perf_counter()

        # -------------------------------------------------
        # COCO EVALUATION
        # -------------------------------------------------
        should_eval = (epoch + 1) % eval_freq == 0 or epoch == int(args.epochs) - 1

        results = None
        if should_eval:

            detector.set_eval_mode()
            metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)

            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Eval mAP]"):
                detector.evaluate(images, targets, metric)

            results = metric.compute()
            current_map = results["map"].item()

            if current_map > best_map:
                best_map = current_map
                epochs_no_improve = 0
                # os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved BEST model to {best_model_path}")

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        # -------------------------------------------------
        # LOGGING
        # -------------------------------------------------
        metrics_to_log = {
            "train/loss": epoch_train_loss / len(train_loader),
            "val/loss": epoch_val_loss / len(val_loader),
            "lr": optimizer.param_groups[0]['lr'],
            "train/time": t_train1 - t_train0,
            "val/time": t_val1 - t_val0,
        }

        if results is not None:
            print(f"mAP@[.5:.95]: {results['map']:.4f}")
            print(f"mAP@50: {results['map_50']:.4f}")
            print(f"mAP@75: {results['map_75']:.4f}")

            metrics_to_log.update({
                "mAP/main": results["map"],
                "mAP/50": results["map_50"],
                "mAP/75": results["map_75"],
                "mAP/small": results["map_small"],
                "mAP/medium": results["map_medium"],
                "mAP/large": results["map_large"],
                "mAR/100": results["mar_100"],
            })

        if args.log_wandb:
            wandb.log(metrics_to_log)

    torch.save(model.state_dict(), final_model_path)
    print(f"Saved FINAL model to {final_model_path}")

    if args.log_wandb:
        wandb.finish()