from functools import partial

import albumentations as A
import torch
import wandb
from huggingface_hub import interpreter_login
from peft import LoraConfig, get_peft_model
from src.custom_datasets import KittiDatasetHuggingface
from src.utils.huggingface_commons import (
    WandbImageLoggerCallback,
    augment_and_transform_batch,
    collate_fn,
    compute_metrics,
    print_trainable_parameters,
)
from transformers import (
    DetrImageProcessorFast,
    AutoModelForObjectDetection,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import time


def train(args):
    interpreter_login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.variant
    dataset = args.dataset
    annotation_folder = args.annotation_folder
    image_folder = args.image_folder
    threshold = args.threshold
    log_wandb = args.log_wandb

    # Load Model
    model = AutoModelForObjectDetection.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=".*decoder.*(_proj|fc.*)",
        lora_dropout=0.1,
        bias="lora_only",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    model.to(device=device)

    # Load Dataset
    train_dataset = KittiDatasetHuggingface(
        dataset, annotation_folder, image_folder, "src/custom_datasets/train.seqmap"
    ).get_hf_ds()
    val_dataset = KittiDatasetHuggingface(
        dataset, annotation_folder, image_folder, "src/custom_datasets/val.seqmap"
    ).get_hf_ds()

    data2model = {1: model.config.label2id["car"], 2: model.config.label2id["person"]}

    id2label = {
        model.config.label2id["car"]: "car",
        model.config.label2id["person"]: "pedestrian",
    }
    # label2id = {v: k for k, v in id2label.items()}

    image_processor = DetrImageProcessorFast.from_pretrained(model_name)

    train_augment_and_transform = A.Compose(
        [
            A.GaussianBlur(sigma_limit=[0.5, 1.0], p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Perspective(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"], clip=True, min_area=25
        ),
    )

    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    train_transform_batch = partial(
        augment_and_transform_batch,
        transform=train_augment_and_transform,
        image_processor=image_processor,
        class_map=data2model,
    )

    validation_transform_batch = partial(
        augment_and_transform_batch,
        transform=validation_transform,
        image_processor=image_processor,
        class_map=data2model,
    )

    train_dataset = train_dataset.with_transform(train_transform_batch)
    val_dataset = val_dataset.with_transform(validation_transform_batch)

    # Define Metrics
    eval_compute_metrics_fn = partial(
        compute_metrics,
        image_processor=image_processor,
        threshold=threshold,
        id2label=id2label,
    )

    # Initialize wandb
    if log_wandb:
        wandb.init(
            project="huggingface",
            entity="c5-team2",
            name=f"Finetune-{args.model}-{args.variant}",
            config=args,
        )

    # Trainer
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    training_args = TrainingArguments(
        output_dir="results/task_e/checkpoints",
        num_train_epochs=epochs,
        fp16=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        eval_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="finetuned-detr",
        eval_do_concat_batches=False,
        push_to_hub=False,
        logging_dir="results/task_e/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            WandbImageLoggerCallback(
                num_samples=4, threshold=threshold, id2label=id2label
            ),
        ],
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    elapsed_total_time = end_time - start_time


    metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
    elapsed_eval_time = metrics["eval_runtime"] * trainer.state.epoch

    # Create a new dictionary for custom W&B names
    metrics_to_log = {
        "mAP/main": metrics["eval_map"],  # AP @.50:.95
        "mAP/50": metrics["eval_map_50"],  # AP @.50
        "mAP/75": metrics["eval_map_75"],  # AP @.75
        "mAP/class_Car": metrics["eval_map_class_Car"],
        "mAP/class_Pedestrian": metrics["eval_map_class_Pedestrian"],
        "mAP/small": metrics["eval_map_small"],
        "mAP/medium": metrics["eval_map_medium"],
        "mAP/large": metrics["eval_map_large"],
        "mAR/Det1": metrics["eval_mar_1"],  # AR
        "mAR/Det10": metrics["eval_mar_10"],
        "mAR/Det100": metrics["eval_mar_100"],
        "mAR/Det100_class_Car": metrics["eval_mar_100_class_Car"],
        "mAR/Det100_class_Pedestrian": metrics["eval_mar_100_class_Pedestrian"],
        "mAR/small": metrics["eval_mar_small"],
        "mAR/medium": metrics["eval_mar_medium"],
        "mAR/large": metrics["eval_mar_large"],
        "performance/total_train_time": elapsed_total_time,
        "performance/time_per_epoch": (elapsed_total_time - elapsed_eval_time) / trainer.state.epoch,
        "performance/total_eval_time": metrics["eval_runtime"],
        "performance/fps": metrics["eval_samples_per_second"],
        "performance/avg_time_per_img": 1 / metrics["eval_samples_per_second"],
    }

    # Log your perfectly formatted metrics directly
    if log_wandb:
        wandb.log(metrics_to_log)
        wandb.finish()

    print(f"\nEvaluation Finished for {args.model}:")
    print(f"mAP @.50:.95:  {metrics['eval_map']:.4f}")
    print(f"mAP @.50:      {metrics['eval_map_50']:.4f}")
    print(f"mAP @.75:      {metrics['eval_map_75']:.4f}")
    print(f"mAR Det1:      {metrics['eval_mar_1']:.4f}")
    print(f"mAR Det10:     {metrics['eval_mar_10']:.4f}")
    print(f"mAR Det100:    {metrics['eval_mar_100']:.4f}")
