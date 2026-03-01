import argparse

from src.task_d import eval_fasterrcnn, eval_huggingface, eval_yolo


def main_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dataset/KITTI-MOTS",
                        help="Path to the root of the dataset. Defaults to dataset/KITTI-MOTS.")
    parser.add_argument('--annotation_folder', type=str, default='instances_txt',
                        help="Name of the folder with the .txt annotations. Defaults to instances_txt.")
    parser.add_argument('--image_folder', type=str, default='training',
                        help="Name of the images folder. Defaults to training.")
    parser.add_argument("--model", type=str, default="fasterrcnn",
                        help="Main model to use on evaluation. fasterrcnn uses Torchvision, detr uses HuggingFace and yolo uses Ultralytics. Defaults to fasterrcnn.")
    parser.add_argument("--variant", type=str, default="resnet50_fpn_v2",
                        help="Variant of the model to use on evaluation. Defaults to resnet50_fpn_v2.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference. Defaults to 8.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for the score of each bounding box. Defaults to 0.5.")
    parser.add_argument(
        "--no_log_wandb", dest="log_wandb", action="store_false",
        help="Disable logging to Weights & Biases"
    )

    parser.set_defaults(log_wandb=True)

    args = parser.parse_args()

    if args.model == "fasterrcnn":
        eval_fasterrcnn(args)

    elif args.model == "detr":
        eval_huggingface(args)

    elif args.model == "yolo":
        eval_yolo(args)


if __name__ == "__main__":
    main_evaluation()
