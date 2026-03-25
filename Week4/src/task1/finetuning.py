
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb

from src.task1.models import load_model_and_processor
from src.task1.dataset import load_annotations, VizWizCaptionDataset, collate_fn


def load_dataset(data_dir: str, processor: object, batch_size: int, num_workers: int):
    """Load the training dataset."""

    data_dir = Path(data_dir)
    samples = load_annotations(data_dir / "annotations" / "train.json")

    dataset = VizWizCaptionDataset(
        data_dir=data_dir,
        samples=samples,
        processor=processor,
        training=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader


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

    for batch in train_loader:

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


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

    # load dataset
    loader = load_dataset(args.data_dir, processor,
                          args.batch_size, args.num_workers)

    # training configs
    optimizer = torch.optim.AdamW([
        {"params": vision_encoder.parameters(), "lr": args.lr_encoder},
        {"params": text_decoder.parameters(), "lr": args.lr_decoder},
    ], weight_decay=args.weight_decay)

    # training loop
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_step(loader, model, optimizer, device)

