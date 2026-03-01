import torch
import wandb
import argparse
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from src.models import FasterRCNN
from src.custom_datasets import KittiDatasetTorchvision
from src.utils import draw_bbox

def evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="C5-Week1", 
        entity="c5-team2", 
        name=f"Eval-{args.model}-{args.variant}",
        config=args
    )

    ds = KittiDatasetTorchvision('dataset/KITTI-MOTS', 'instances_txt', 'training', 'src/custom_datasets/val.seqmap')
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=lambda x: tuple(zip(*x)))
    detector = FasterRCNN(variant=args.variant, threshold=args.threshold, device=device)
    detector.set_eval_mode()
    # class_metrics=True to see mAP for Car and Pedestrian separately
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    max_images_to_log = 8
    logged = False
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Evaluating")):
        detector.evaluate(images, targets, metric)

        if not logged:
 
            preds = detector.inference(images)

            # id2label 
            id2label_coco = {i: name for i, name in enumerate(detector.categories)}
            id2label_kitti = {1: "Car", 2: "Pedestrian"}

            images_to_log = []
            for i in range(min(len(images), max_images_to_log)):

                img = images[i].detach().cpu()

                gt_dict = {
                    "boxes": targets[i]["boxes"].detach().cpu(),
                    "labels": targets[i]["labels"].detach().cpu().to(torch.int64),
                }

                p = preds[i]

                if detector.kitti_mapping is not None:
                    keep = [j for j, lab in enumerate(p["labels"]) if lab.item() in detector.kitti_mapping]

                    pred_dict = {
                        "boxes": p["boxes"][keep].detach().cpu(),
                        "scores": p["scores"][keep].detach().cpu(),
                        "labels": torch.tensor(
                            [detector.kitti_mapping[lab.item()] for lab in p["labels"][keep]],
                            dtype=torch.int64
                        )
                    }
                    pred_id2label = id2label_kitti
                else:
                    pred_dict = {
                        "boxes": p["boxes"].detach().cpu(),
                        "scores": p["scores"].detach().cpu(),
                        "labels": p["labels"].detach().cpu().to(torch.int64),
                    }
                    pred_id2label = id2label_coco

                # dibujar GT rojo + pred verde
                drawing = draw_bbox(
                    img,
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
                    id2label=pred_id2label,
                    format="pascal_voc",
                    boxes_key="boxes",
                    labels_key="labels",
                    scores_key="scores",
                    colors="green",
                )

                images_to_log.append(
                    wandb.Image(drawing, caption=f"Eval batch {batch_idx} sample {i} (red=GT, green=Pred)")
                )

            wandb.log({"eval/visualizations": images_to_log})
            logged = True

    
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
    
    wandb.log(metrics_to_log)

    print(f"\nEvaluation Finished for {args.variant}:")
    print(f"mAP @.50:.95: {results['map']:.4f}")
    print(f"mAP @.50:      {results['map_50']:.4f}")
    print(f"mAP @.75:      {results['map_75']:.4f}")
    print(f"mAR Det1:      {results['mar_1']:.4f}")
    print(f"mAR Det10:     {results['mar_10']:.4f}")
    print(f"mAR Det100:    {results['mar_100']:.4f}")


    wandb.finish()

