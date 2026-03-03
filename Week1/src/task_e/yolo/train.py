import torch
import albumentations as A
import yaml
import time

from src.models import YOLOModel

from dotenv import load_dotenv
load_dotenv()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = f"{args.variant}_{args.augmentation[:7]}_"

    if args.augmentation == "albumentations":
        cfg = "/ghome/group02/Marina/C5-Project/Week1/config/yolo_train_augmentation.yaml"
        # Training with custom Albumentations transforms
        custom_transforms = [
            A.GaussianBlur(sigma_limit=[0.5, 1.0], p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Perspective(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ]

    elif args.augmentation == "default":
        cfg = "/ghome/group02/Marina/C5-Project/Week1/config/yolo_train_augmentation.yaml"

    elif args.augmentation == "none" or not args.augmentation:
        cfg = "/ghome/group02/Marina/C5-Project/Week1/config/yolo_train_no_augmentation.yaml"

    else:
        raise ValueError(
            f"Augmentation argument value {args.augmentation} not valid.")

    detector = YOLOModel(model=args.variant, device=device)

    # configure the layers to freeze in the training
    with open(cfg, 'r') as f:
        configs = yaml.safe_load(f)

    with open(cfg, 'w') as f:
        configs["freeze"] = args.unfreeze_depth
        name += f"freeze{args.unfreeze_depth}"
        configs["name"] = name
        yaml.safe_dump(configs, f, sort_keys=False)

    start_time = time.time()

    if args.augmentation == "albumentations":
        detector.train(cfg=cfg, augmentations=custom_transforms)
    else:
        detector.train(cfg=cfg)

    end_time = time.time()
    elapsed_total_time = end_time - start_time

    total_params = sum(p.numel() for p in detector.model.parameters())
    trainable_params = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Ellapsed time during training: {elapsed_total_time:.2f} seconds")
    print(f"Training time per epoch: {elapsed_total_time / detector.model.trainer.epoch + 1} seconds")
    
