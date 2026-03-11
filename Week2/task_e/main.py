import argparse
import sys
from pathlib import Path

import torch
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader

from train import train


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK2_SRC = PROJECT_ROOT / "Week2" / "src"

if str(WEEK2_SRC) not in sys.path:
    sys.path.append(str(WEEK2_SRC))

from utils.kitti_dataset_motsio2 import KittiDataset


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_argparser():
    parser = argparse.ArgumentParser(description="Fine-tune SAM on KITTI-MOTS.")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "data" / "KITTI-MOTS" / "training",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "data" / "KITTI-MOTS" / "instances_txt",
    )
    parser.add_argument(
        "--train-seqmap",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "src" / "utils" / "train.seqmap",
    )
    parser.add_argument(
        "--val-seqmap",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "src" / "utils" / "val.seqmap",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--model-type", type=str, default="vit_b")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Week2" / "task_e" / "checkpoints",
    )
    parser.add_argument("--train-image-encoder", action="store_true")
    parser.add_argument("--train-prompt-encoder", action="store_true")

    return parser


def resolve_checkpoint(checkpoint_arg: Path | None) -> Path:
    if checkpoint_arg is not None and checkpoint_arg.exists():
        return checkpoint_arg.resolve()

    candidate_paths = [
        PROJECT_ROOT / "Week2" / "task_e" / "sam_vit_b.pth",
        PROJECT_ROOT / "Week2" / "task_e" / "sam_vit_b_01ec64.pth",
        PROJECT_ROOT / "sam_vit_b.pth",
        PROJECT_ROOT / "sam_vit_b_01ec64.pth",
        Path.cwd() / "sam_vit_b.pth",
        Path.cwd() / "sam_vit_b_01ec64.pth",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()

    searched_paths = "\n".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        "SAM checkpoint not found.\n"
        "Expected a local checkpoint such as `sam_vit_b_01ec64.pth` or `sam_vit_b.pth`.\n"
        f"Searched:\n{searched_paths}\n"
        "Pass `--checkpoint /absolute/path/to/sam_vit_b_01ec64.pth` if it is stored elsewhere."
    )


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    train_dataset = KittiDataset(
        str(args.data_root),
        str(args.annotations_root),
        str(args.train_seqmap),
        only_mask=False,
    )
    val_dataset = KittiDataset(
        str(args.data_root),
        str(args.annotations_root),
        str(args.val_seqmap),
        only_mask=False,
    )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint_path))
    sam.to(device)

    if not args.train_image_encoder:
        for param in sam.image_encoder.parameters():
            param.requires_grad = False

    if not args.train_prompt_encoder:
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False

    trainable_params = [param for param in sam.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Enable at least one SAM component.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Starting fine-tuning...")
    print(f"Trainable parameter tensors: {len(trainable_params)}")
    print(f"Using checkpoint: {checkpoint_path}")

    train(
        model=sam,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
