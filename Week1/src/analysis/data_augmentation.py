import albumentations as A
import argparse
from pathlib import Path
from torchvision.transforms import ToPILImage
import os
from src.utils.drawing import draw_bbox
from src.custom_datasets import KittiDatasetTorchvision
import numpy as np
from PIL import Image
from tqdm import tqdm

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folder', type=str)
    parser.add_argument('--dataset', type=str, default="dataset/KITTI-MOTS",
                        help="Path to the root of the dataset. Defaults to dataset/KITTI-MOTS.")
    parser.add_argument('--annotation_folder', type=str, default='instances_txt',
                        help="Name of the folder with the .txt annotations. Defaults to instances_txt.")
    parser.add_argument('--image_folder', type=str, default='training',
                        help="Name of the images folder. Defaults to training.")
    parser.add_argument('--seqmap', type=str, default='src/custom_datasets/val.seqmap')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(args.save_folder, exist_ok=True)
    folder = Path(args.save_folder)

    transforms = [
        (A.GaussianBlur(sigma_limit=[0.5, 1.0], p=1.0), 'gaussian'),
        (A.HorizontalFlip(p=1.0), 'hflip'),
        (A.RandomBrightnessContrast(p=1.0), 'randbright'),
        (A.Perspective(p=1.0), 'persp'),
        (A.HueSaturationValue(p=1.0), 'hue')
    ]
    control_dataset = KittiDatasetTorchvision(args.dataset, args.annotation_folder, args.image_folder, args.seqmap)

    for transform, prefix in tqdm(transforms):
        t = A.Compose([transform], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), seed=args.seed)
        dataset = KittiDatasetTorchvision(args.dataset, args.annotation_folder, args.image_folder, args.seqmap, transform=t)

        for i in range(args.num_samples):
            # Define image path
            file_name = f"{prefix}{i:04d}.png"
            file_path = folder/file_name

            # Get image
            gt_img, gt_bboxes = control_dataset[i]
            gt_img = gt_img.permute(1, 2, 0).numpy()
            augmented_img, augmented_bboxes = dataset[i]
            augmented_img = augmented_img.permute(1, 2, 0).numpy()

            # Mosaic for comparison
            id2label = {1: 'car', 2: 'pedestrian'}
            gt_annotated = draw_bbox(gt_img, gt_bboxes, id2label, scores_key='area')
            augmented_annotated = draw_bbox(augmented_img, augmented_bboxes, id2label, scores_key='area')

            axis = 1 if gt_img.shape[0] > gt_img.shape[1] else 0
            comparison = np.concatenate([gt_annotated, augmented_annotated], axis=axis)

            comparison = Image.fromarray(comparison)
            comparison.save(file_path)

if __name__ == '__main__':
    args = args_parser()
    main(args)