from functools import partial

import torch
import wandb
from huggingface_hub import interpreter_login
from src.custom_datasets import KittiDatasetHuggingface
from src.utils.huggingface_commons import (
    load_model,
    WandbImageLoggerCallback,
    augment_and_transform_batch,
    collate_fn,
    compute_metrics,
)
from transformers import (
    DetrImageProcessorFast,
    Trainer,
    TrainingArguments,
)


def evaluation(args):
    interpreter_login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.variant
    dataset = args.dataset
    annotation_folder = args.annotation_folder
    image_folder = args.image_folder
    batch_size = args.batch_size
    threshold = args.threshold
    log_wandb = args.log_wandb
    lora = args.lora

    # Load Model
    model = load_model(model_name, lora_path=lora, merged=True)
    model.to(device)

    # Load Dataset
    dataset = KittiDatasetHuggingface(
        dataset, annotation_folder, image_folder, "src/custom_datasets/val.seqmap"
    ).get_hf_ds()

    data2model = {1: model.config.label2id["car"], 2: model.config.label2id["person"]}

    id2label = {
        model.config.label2id["car"]: "car",
        model.config.label2id["person"]: "pedestrian",
    }
    # label2id = {v: k for k, v in id2label.items()}

    image_processor = DetrImageProcessorFast.from_pretrained(model_name)

    validation_transform_batch = partial(
        augment_and_transform_batch,
        image_processor=image_processor,
        class_map=data2model,
        transform=None,
    )

    dataset = dataset.with_transform(validation_transform_batch)

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
            name=f"Eval-{args.model}-{args.variant}",
            config=args,
        )

    # Trainer
    training_args = TrainingArguments(
        output_dir="results/task_d",
        num_train_epochs=30,
        fp16=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        run_name="pretrained-detr-eval",
        eval_do_concat_batches=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
        callbacks=[
            WandbImageLoggerCallback(
                num_samples=4, threshold=threshold, id2label=id2label
            )
        ],
    )

    metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix="eval")

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
        "performance/total_time": metrics["eval_runtime"],
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
