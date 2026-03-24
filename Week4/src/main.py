import argparse
from src.task1 import run_inference
from src.utils import load_config

def args_parser():
    main_parser = argparse.ArgumentParser(description="C5 Week4 - Image Captioning")
    subparsers = main_parser.add_subparsers(required=True)

    # Inference 
    infer_parser = subparsers.add_parser("inference", help="Run Inference")
    infer_parser.add_argument("--config", required=True)
    infer_parser.set_defaults(func=run_inference)

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
