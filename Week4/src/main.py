import argparse
from src.task1 import run_inference1, run_finetuning1
from src.task2 import run_inference2, run_finetuning2
from src.utils import load_config


def args_parser():
    main_parser = argparse.ArgumentParser(
        description="C5 Week4 - Image Captioning")
    subparsers = main_parser.add_subparsers(required=True)

    # Inference
    infer_parser = subparsers.add_parser("inference1", help="Run Inference of task 1")
    infer_parser.add_argument("--config", required=True)
    infer_parser.set_defaults(func=run_inference1)

    # Finetuning
    finetune_parser = subparsers.add_parser(
        "finetuning1", help="Run Finetuning of task 1")
    finetune_parser.add_argument("--config", required=True)
    finetune_parser.set_defaults(func=run_finetuning1)
    
    # Inference
    infer_parser = subparsers.add_parser("inference2", help="Run Inference of task 2")
    infer_parser.add_argument("--config", required=True)
    infer_parser.set_defaults(func=run_inference2)

    # Finetuning
    finetune_parser = subparsers.add_parser(
        "finetuning2", help="Run Finetuning of task 2")
    finetune_parser.add_argument("--config", required=True)
    finetune_parser.set_defaults(func=run_finetuning2)

    return main_parser.parse_args()


def main(args):
    if hasattr(args, "config") and args.config:
        yaml_config = load_config(args.config)
        for key, value in yaml_config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    args.func(args)


if __name__ == "__main__":
    args = args_parser()
    main(args)
