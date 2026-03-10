import argparse
import yaml
from src.task_c.inference_sam_bbox import main_task_c

def parse_config(parser: argparse.ArgumentParser, config_file: str):
    """
    Overwrites the argument with those of the given config_file if any.
    """
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)
    parser.set_defaults(**yaml_config)
    args = parser.parse_args()
    return args


def args_parser():
    """
    Parses the arguments of the script.
    """
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file with all the arguments configuration.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Entry point to run a specific part of the repo.",
    )

    # Task N

    # Task W

    args = parser.parse_args()

    if args.config:
        args = parse_config(parser, args.config)

    return args


def main(args):
    """
    Entry point of every task
    """
    task = args.task
    assert task

    if task == "task_c":
        main_task_c(args)


if __name__ == "__main__":
    args = args_parser()
    main(args)
