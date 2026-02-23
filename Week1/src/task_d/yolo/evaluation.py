import torch
import wandb
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from src.models import YOLOModel
from src.custom_datasets import KittiDatasetUltralytics

from dotenv import load_dotenv
load_dotenv()


def evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.log_wandb:
        wandb.init(
            project="C5-Week1",
            entity="c5-team2",
            name=f"Eval-{args.model}-{args.variant}",
            config=args
        )

    ds = KittiDatasetUltralytics('dataset/KITTI-MOTS', 'instances_txt',
                                 'training', 'src/custom_datasets/val.seqmap')
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=lambda x: tuple(zip(*x)))
    detector = YOLOModel(model=args.variant, device=device)
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    for images, targets in tqdm(loader, desc="Evaluating"):
        detector.evaluate(images, targets, metric)

    print("Computing final metrics...")
    results = metric.compute()

    metrics_to_log = {
        "mAP/main": results["map"],          # AP @.50:.95
        "mAP/50": results["map_50"],         # AP @.50
        "mAP/75": results["map_75"],         # AP @.75
        "mAP/small": results["map_small"],
        "mAP/medium": results["map_medium"],
        "mAP/large": results["map_large"],

        "mAR/Det1": results["mar_1"],        # AR
        "mAR/Det10": results["mar_10"],
        "mAR/Det100": results["mar_100"],
        "mAR/small": results["mar_small"],
        "mAR/medium": results["mar_medium"],
        "mAR/large": results["mar_large"],
    }

    if "map_per_class" in results:
        # 1: Car, 2: Pedestrian
        metrics_to_log["mAP/class_Car"] = results["map_per_class"][0]
        metrics_to_log["mAP/class_Pedestrian"] = results["map_per_class"][1]

    if "mar_100_per_class" in results:
        metrics_to_log["mAR/Det100_class_Car"] = results["mar_100_per_class"][0]
        metrics_to_log["mAR/Det100_class_Pedestrian"] = results["mar_100_per_class"][1]

    if args.log_wandb:
        wandb.log(metrics_to_log)

    print(f"\nEvaluation Finished for {args.variant}:")
    print(f"mAP @.50:.95:  {results['map']:.4f}")
    print(f"mAP @.50:      {results['map_50']:.4f}")
    print(f"mAP @.75:      {results['map_75']:.4f}")
    print(f"mAR Det1:      {results['mar_1']:.4f}")
    print(f"mAR Det10:     {results['mar_10']:.4f}")
    print(f"mAR Det100:    {results['mar_100']:.4f}")

    if args.log_wandb:
        wandb.finish()
