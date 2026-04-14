import argparse
from src.task1.diffusion_inference import run_diffusion_inference

from src.utils import load_config

from dotenv import load_dotenv

load_dotenv()


def args_parser():
    main_parser = argparse.ArgumentParser(
        description="C5 Week5 - Diffusion Inference")
    main_parser.add_argument("--config", required=True,
                             help="Path to the config file")
    subparsers = main_parser.add_subparsers(required=True)

    # Diffusion Inference specific arguments
    diffusion_inference_parser = subparsers.add_parser(
        "diffusion_inference", help="Run Diffusion Inference")
    diffusion_inference_parser.add_argument(
        "--model_name", type=str, default=None,
        help="Name of the diffusion model to use")
    diffusion_inference_parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt for image generation")
    diffusion_inference_parser.add_argument(
        "--num_inference_steps", type=int, default=None,
        help="Number of inference steps for the diffusion model")
    diffusion_inference_parser.add_argument(
        "--image_prompt", type=str, default=None,
        help="Path to an image prompt for image-to-image generation")
    diffusion_inference_parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed used to reproduce generations")
    diffusion_inference_parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory where generated images will be stored")
    diffusion_inference_parser.set_defaults(func=run_diffusion_inference)

    return main_parser.parse_args()


def main(args):
    if hasattr(args, "config") and args.config:
        yaml_config = load_config(args.config)
        for key, value in yaml_config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    args.func(args)


if __name__ == "__main__":
    args = args_parser()
    main(args)
