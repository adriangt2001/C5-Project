import argparse

# from src.task_e import train_fasterrcnn, train_huggingface, train_yolo
from src.task_e import train_yolo, train_huggingface, train_fasterrcnn


def main_training():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/KITTI-MOTS",
        help="Path to the root of the dataset. Defaults to dataset/KITTI-MOTS.",
    )
    parser.add_argument(
        "--annotation_folder",
        type=str,
        default="instances_txt",
        help="Name of the folder with the .txt annotations. Defaults to instances_txt.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="training",
        help="Name of the images folder. Defaults to training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fasterrcnn",
        help="Main model to use on evaluation. fasterrcnn uses Torchvision, detr uses HuggingFace and yolo uses Ultralytics. Defaults to fasterrcnn.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="resnet50_fpn_v2",
        help="Variant of the model to use on evaluation. Defaults to resnet50_fpn_v2.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference. Defaults to 8.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for the score of each bounding box. Defaults to 0.5.",
    )
    parser.add_argument(
        "--no_log_wandb",
        dest="log_wandb",
        action="store_false",
        help="Disable logging to Weights & Biases",
    )

    # Extras for training
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Starting learning rate for training. Defaults to 5e-4.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=50,
        help="Max num. epochs for training. Defaults to 50.",
    )
    parser.add_argument(
        "--unfreeze_depth",
        type=int,
        default=0,
        help="Number of layers to unfreeze. Defaults to 0.",
    )
    parser.add_argument(
        "--lora_layer",
        type=str,
        choices=["prediction", "decoder", "encoder"],
        default="prediction",
        help="Layers to apply LoRA if LoRA is used. It's cummulative, decoder includes prediction and encoder includes decoder and prediction.",
    )

    parser.set_defaults(log_wandb=True)

    args = parser.parse_args()

    if args.model == "fasterrcnn":
        train_fasterrcnn(args)

    elif args.model == "detr":
        train_huggingface(args)

    elif args.model == "yolo":
        train_yolo(args)


if __name__ == "__main__":
    main_training()
