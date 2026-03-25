from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb

from src.task1.models import load_model_and_processor
from src.task1.dataset import (
    load_annotations,
    split_train_val,
    VizWizCaptionDataset,
    build_train_collate_fn,
    collate_fn,
)
from src.utils import compute_metrics


def build_dataloaders(args, processor):
    """Build train and validation dataloaders for finetuning."""

    data_dir = Path(args.data_dir)
    samples = load_annotations(data_dir / "annotations" / "train.json")
    train_samples, val_samples = split_train_val(
        samples,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )

    max_len = min(processor.tokenizer.model_max_length, 40)

    train_dataset = VizWizCaptionDataset(
        data_dir=data_dir,
        samples=train_samples,
        processor=processor,
        max_len=max_len,
        training=True,
    )
    val_dataset = VizWizCaptionDataset(
        data_dir=data_dir,
        samples=val_samples,
        processor=processor,
        max_len=max_len,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_train_collate_fn(
            processor=processor,
            max_len=train_dataset.max_len,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def set_finetuning(args, model):

    # access encoder and decoder
    if args.model_type == "blip":
        vision_encoder = model.vision_model
        text_decoder = model.text_decoder
    elif args.model_type == "vit-gpt2":
        raise NotImplementedError(
            "Finetuning not implemented for vit-gpt2 yet")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    if not args.finetune_encoder and not args.finetune_decoder:
        raise ValueError(
            "At least one of finetune_encoder or finetune_decoder must be True")

    # setting finetuning
    if args.finetune_encoder:
        print("Finetuning encoder...")
        for p in vision_encoder.parameters():
            p.requires_grad = True
    else:
        print("Freezing encoder...")
        for p in vision_encoder.parameters():
            p.requires_grad = False

    if args.finetune_decoder:
        print("Finetuning decoder...")
        for p in text_decoder.parameters():
            p.requires_grad = True
    else:
        print("Freezing decoder...")
        for p in text_decoder.parameters():
            p.requires_grad = False

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    encoder_trainable_params = sum(
        parameter.numel()
        for parameter in vision_encoder.parameters()
        if parameter.requires_grad
    )
    decoder_trainable_params = sum(
        parameter.numel()
        for parameter in text_decoder.parameters()
        if parameter.requires_grad
    )

    print(
        "model_params "
        f"total={total_params} "
        f"trainable={trainable_params} "
        f"encoder_trainable={encoder_trainable_params} "
        f"decoder_trainable={decoder_trainable_params}"
    )

    return vision_encoder, text_decoder


def train_step(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


def evaluate_step(model, processor, val_loader, args, device):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            for pred, refs, fname, img_id in zip(
                predictions,
                batch["references"],
                batch["file_names"],
                batch["image_ids"],
            ):
                results.append({
                    "image_id": img_id,
                    "file_name": fname,
                    "prediction": pred.strip(),
                    "references": refs,
                })

    return compute_metrics(results)


def run_finetuning(args):

    wandb_cfg = args.wandb
    if wandb_cfg["enabled"]:
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=wandb_cfg["name"],
            config=vars(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and processor
    model, processor, tokenizer = load_model_and_processor(
        args.model_type, args.model_name, device, mode="finetuning")

    vision_encoder, text_decoder = set_finetuning(args, model)

    # load datasets
    train_loader, val_loader = build_dataloaders(args, processor)

    # training configs
    param_groups = []
    if args.finetune_encoder:
        param_groups.append({
            "params": [p for p in vision_encoder.parameters() if p.requires_grad],
            "lr": args.lr_encoder,
        })
    if args.finetune_decoder:
        param_groups.append({
            "params": [p for p in text_decoder.parameters() if p.requires_grad],
            "lr": args.lr_decoder,
        })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    print("Computing baseline validation metrics before training...")
    baseline_metrics = evaluate_step(model, processor, val_loader, args, device)
    print("\n--- Baseline Validation Metrics ---")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    if wandb_cfg["enabled"]:
        wandb.log(
            {f"val/{k}": v for k, v in baseline_metrics.items()} |
            {"epoch": 0}
        )

    # training loop
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_loss = train_step(train_loader, model, optimizer, device)
        print(f"Epoch {epoch} train_loss: {train_loss:.4f}")

        print(f"Computing validation metrics for epoch {epoch}...")
        metrics = evaluate_step(model, processor, val_loader, args, device)
        print("\n--- Validation Metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if wandb_cfg["enabled"]:
            wandb.log(
                {"train/loss": train_loss} |
                {f"val/{k}": v for k, v in metrics.items()} |
                {"epoch": epoch}
            )

    if wandb_cfg["enabled"]:
        wandb.finish()
