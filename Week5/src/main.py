import argparse

from src.task_a import run_task_a
from src.task_b import run_task_b
from src.task_d import run_task_d_flux, run_build_synthetic_annotations
from src.task_e import run_finetuning
from src.utils import load_config


def build_parser():
    parser = argparse.ArgumentParser(description="C5 Week5 - Diffusion models")
    subparsers = parser.add_subparsers(required=True)

    # Task a: Compare open diffusion models
    task_a_parser = subparsers.add_parser(
        "task_a", help="Compare open diffusion models")
    task_a_parser.add_argument("--config", required=True)
    task_a_parser.set_defaults(func=run_task_a)

    # Task b: Inference sweep
    task_b_parser = subparsers.add_parser("task_b", help="Run inference sweep")
    task_b_parser.add_argument("--config", required=True)
    task_b_parser.set_defaults(func=run_task_b)

    # Task d: Generate images with FLUX.2-dev-bnb-4bit
    task_d_flux_parser = subparsers.add_parser(
        "task_d_flux", help="Generate images with FLUX.2-dev-bnb-4bit")
    task_d_flux_parser.add_argument("--config", required=True)
    task_d_flux_parser.add_argument(
        "--model_id", type=str, default="diffusers/FLUX.2-dev-bnb-4bit")
    task_d_flux_parser.add_argument("--prompts", nargs="*", default=None)
    task_d_flux_parser.add_argument("--prompts_json", type=str, default=None)
    task_d_flux_parser.add_argument("--output_dir", type=str,
                                    default=str("Week5/results/flux"))
    task_d_flux_parser.add_argument("--report_path", type=str, default=None)
    task_d_flux_parser.add_argument("--device", type=str, default="auto")
    task_d_flux_parser.add_argument("--seed", type=int, default=42)
    task_d_flux_parser.add_argument(
        "--num_inference_steps", type=int, default=60)
    task_d_flux_parser.add_argument(
        "--guidance_scale", type=float, default=3.5)
    task_d_flux_parser.add_argument("--height", type=int, default=1024)
    task_d_flux_parser.add_argument("--width", type=int, default=1024)
    task_d_flux_parser.add_argument("--limit", type=int, default=None)
    task_d_flux_parser.add_argument("--skip_existing", action="store_true")
    task_d_flux_parser.set_defaults(func=run_task_d_flux)

    # Task e: Finetune captioning model with augmented data
    synthetic_annotations_parser = subparsers.add_parser(
        "build_synthetic_annotations", help="Build train_synthetic.json from generated images"
    )
    synthetic_annotations_parser.add_argument("--config", required=True)
    synthetic_annotations_parser.add_argument(
        "--limit", type=int, default=None)
    synthetic_annotations_parser.set_defaults(
        func=run_build_synthetic_annotations)

    finetune_parser = subparsers.add_parser(
        "finetuning", help="Run Finetuning")
    finetune_parser.add_argument("--config", required=True)
    finetune_parser.set_defaults(func=run_finetuning)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    for key, value in config.items():
        if key in {"config", "func"}:
            continue
        setattr(args, key, value)

    args.func(args)


if __name__ == "__main__":
    main()
