import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from datasets import Image as DImage
from huggingface_hub import interpreter_login
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from .dataset_base import KittiDataset


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default="microsoft/conditional-detr-resnet-50")
    parser.add_argument('--annotations_folder', type=str, default='instances_txt')
    parser.add_argument('--image_folder', type=str, default='training')
    parser.add_argument('--results', type=str, default='results')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    assert os.path.exists(args.dataset)
    assert os.path.exists(args.results)

    return args

def create_dataset(root: str, annotations_folder: str, image_folder: str, seqmap_file: str):
    base_ds = KittiDataset(root, annotations_folder, image_folder, seqmap_file)
    features = base_ds.features
    features['image_id'] = list(range(len(features['image'])))
    dataset = Dataset.from_dict(features).cast_column('image', DImage())

    print(dataset)
    return dataset

def save_image_bbox(image: Image.Image, info_list: list, save_folder: str, name: str):
    W, H = image.size
    draw = ImageDraw.Draw(image)
    for info in info_list:
        box = info['box']
        label = info['label']
        score = info['score']
        x1, y1, x2, y2 = tuple(box.values())
        if max(box.values()) < 1.0:
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=1)
        draw.text((x1, y1), f'{label} {score}', fill="white")
    image.save(Path(save_folder) / name)


def main(args):
    interpreter_login()

    IMAGE_SIZE = 480

    pipe = pipeline(
        "object-detection",
        model=args.model,
        dtype=torch.float16,
        device_map=0
    )
    dataset = create_dataset(args.dataset, args.annotations_folder, args.image_folder, 'val.seqmap')

    for idx, out in tqdm(enumerate(pipe(KeyDataset(dataset, "image"))), total=len(dataset)):
        scores = [o['score'] for o in out]

        if idx % 100 == 0 or len(scores) == 0 or min(scores) < args.threshold:
            if idx % 100 == 0:
                reason = ''
            elif len(scores) == 0:
                reason = '_nodet'
            else:
                reason = '_lowconf'
            image = dataset['image'][idx]
            save_image_bbox(image, out, args.results, f'{idx:06d}{reason}.png')

if __name__ == '__main__':
    args = argument_parser()
    main(args)