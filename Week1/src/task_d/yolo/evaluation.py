import torch
import wandb
from tqdm import tqdm
import time

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from src.models import YOLOModel
from src.custom_datasets import KittiDatasetUltralytics
from src.utils import draw_bbox

from dotenv import load_dotenv
load_dotenv()


def draw_predicted_gt_bboxes(image, target, pred, images_to_log, batch_idx):

    # id2label
    id2label_kitti = {1: "Car", 2: "Pedestrian"}
    allowed_labels = torch.tensor([1, 2], dtype=torch.int64)

    gt_dict = {
        "boxes": target["boxes"].detach().cpu(),
        "labels": target["labels"].detach().cpu().to(torch.int64),
    }

    # Fine-tuned => predictions are already KITTI ids (1,2)
    pred_dict = {
        "boxes": pred["boxes"].detach().cpu(),
        "scores": pred["scores"].detach().cpu(),
        "labels": pred["labels"].detach().cpu().to(torch.int64),
    }

    # Optional safety filter: keep only {1,2}
    keep = torch.isin(pred_dict["labels"], allowed_labels)
    pred_dict = {
        "boxes": pred_dict["boxes"][keep],
        "scores": pred_dict["scores"][keep],
        "labels": pred_dict["labels"][keep],
    }

    drawing = draw_bbox(
        image,
        gt_dict,
        id2label=id2label_kitti,   # GT es KITTI ids
        format="pascal_voc",
        boxes_key="boxes",
        labels_key="labels",
        scores_key=None,
        colors="red",
    )
    drawing = draw_bbox(
        drawing,
        pred_dict,
        id2label=id2label_kitti,
        format="pascal_voc",
        boxes_key="boxes",
        labels_key="labels",
        scores_key="scores",
        colors="green",
    )

    image_to_log = wandb.Image(
        drawing, caption=f"Eval batch {batch_idx} sample 1 (red=GT, green=Pred)")

    wandb.log({"eval/visualizations": image_to_log})


def evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.log_wandb:
        variant_name = args.variant.split("/")[-1]
        wandb.init(
            project="yolo-eval",
            entity="c5-team2",
            name=f"Eval-{args.model}-{variant_name}",
            config=args
        )

    ds = KittiDatasetUltralytics('dataset/KITTI-MOTS', 'instances_txt',
                                 'training', 'src/custom_datasets/val.seqmap')
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=lambda x: tuple(zip(*x)))
    detector = YOLOModel(model=args.variant, device=device)
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    total_time = 0
    total_images = 0

    images_to_log = []

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Evaluating")):
        start_batch = time.perf_counter()

        preds = detector.evaluate(images, targets, metric)

        end_batch = time.perf_counter()

        total_time += (end_batch - start_batch)
        total_images += len(images)

        # dibujar GT rojo + pred verde
        draw_predicted_gt_bboxes(
            images[1], targets[1], preds[1], images_to_log, batch_idx)

    avg_time_per_img = total_time / total_images
    fps = 1 / avg_time_per_img  # frames per second

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

        "performance/avg_time_per_img": avg_time_per_img,
        "performance/fps": fps,
        "performance/total_time": total_time
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
