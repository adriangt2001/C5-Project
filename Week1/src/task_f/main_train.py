import argparse
from .fasterrcnn.train import train

def main_training():

    parser = argparse.ArgumentParser()

    # -------------------------------------------------
    # DATASET
    # -------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to DEArt_dataset root folder"
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    parser.add_argument(
        "--variant",
        type=str,
        default="resnet50_fpn",
        help="Faster R-CNN backbone variant"
    )

    parser.add_argument(
        "--aug",
        type=str,
        default="none",
        choices=["none", "mild", "aggressive"],
        help="Augmentation level"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for fine-tuning"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone weights"
    )

    parser.add_argument(
        "--no_log_wandb",
        dest="log_wandb",
        action="store_false",
        help="Disable Weights & Biases logging"
    )

    parser.set_defaults(log_wandb=True)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main_training()