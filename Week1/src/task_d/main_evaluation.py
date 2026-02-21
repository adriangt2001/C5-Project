import argparse
from src.task_d.fasterrcnn.evaluation import evaluation as eval_fasterrcnn
from src.task_d.yolo.evaluation import evaluation as eval_yolo


def main_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fasterrcnn")
    parser.add_argument("--variant", type=str, default="resnet50_fpn_v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--no_log_wandb", dest="log_wandb", action="store_false",
        help="Disable logging to Weights & Biases"
    )

    parser.set_defaults(log_wandb=True)

    args = parser.parse_args()

    if args.model == "fasterrcnn":
        eval_fasterrcnn(args)

    elif args.model == "detr":
        pass

    elif args.model == "yolo":
        eval_yolo(args)


if __name__ == "__main__":
    main_evaluation()
