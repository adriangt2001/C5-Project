import argparse
import yaml

import sys
sys.path.append("/DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week2/")

from src.task_a.run_task_a import main_task_a
from src.task_b.run_task_b import main_task_b
from src.task_c.inference_sam_bbox import main_task_c
from src.task_f.evaluate_domain_shift import main_task_f
from src.task_g.analyze_prompt_robustness import main_task_g
from src.task_h.inference_semantic_text import main_task_h

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

    # Shared dataset arguments
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week2/data/KITTI-MOTS/training",
    )
    parser.add_argument(
        "--annotations_folder",
        type=str,
        default="/DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week2/data/KITTI-MOTS/instances_txt",
    )
    parser.add_argument(
        "--seqmap_file",
        type=str,
        default="/DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week2/src/utils/val.seqmap",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_visualizations",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="week2-run",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default="product",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week2/src/task_c/fasterrcnn_kittimots_detections.csv",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="baseline",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
    )

    # Task A
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week2/src/results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Number of dataset samples to evaluate for Task A. Use -1 for all samples.",
    )
    parser.add_argument(
        "--prompt_comparison_index",
        type=int,
        default=880,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

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

    if task == "task_a":
        print("Running Task A: SAM point-prompt evaluation")
        main_task_a(args)

    if task == "task_b":
        print("Running Task B: GroundingDINO prompt evaluation")
        main_task_b(args)

    if task == "task_c":
        main_task_c(args)

    if task == "task_f":
        print("Running Task F: Evaluating Domain Shift")
        main_task_f(args)

    if task == "task_g":
        print("Running Task G: Prompt Robustness Analysis")
        main_task_g(args)

    if task == "task_h":
        main_task_h(args)


if __name__ == "__main__":
    args = args_parser()
    main(args)
