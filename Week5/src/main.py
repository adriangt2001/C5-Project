import argparse

from src.task_a import run_task_a
from src.task_b import run_task_b
from src.utils import load_config


def build_parser():
    parser = argparse.ArgumentParser(description="C5 Week5 - Diffusion models")
    subparsers = parser.add_subparsers(required=True)

    task_a_parser = subparsers.add_parser("task_a", help="Compare open diffusion models")
    task_a_parser.add_argument("--config", required=True)
    task_a_parser.set_defaults(func=run_task_a)

    task_b_parser = subparsers.add_parser("task_b", help="Run inference sweep")
    task_b_parser.add_argument("--config", required=True)
    task_b_parser.set_defaults(func=run_task_b)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    args.func(args)


if __name__ == "__main__":
    main()
