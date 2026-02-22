from pathlib import Path
from typing import Literal

import pycocotools.mask as rletools
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import glob
from . import motsio


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, annotations_folder: str, image_folder: str, seqmap_file: str, transforms = None):
        self.transforms = transforms
        root = Path(root)

        self.features = {
            'image_id': [],
            'image': [],
            'width': [],
            'height': [],
            'objects': []
        }
        seqmap, max_frames = motsio.load_seqmap(seqmap_file)

        for seq in seqmap:        
            self.load_sequence(root/annotations_folder/f"{seq}.txt", root/image_folder/'image_02'/seq)

    def load_sequence(self, txt_path, image_folder):
        txt_path = Path(txt_path)
        image_folder = Path(image_folder)
        loaded_txt = motsio.load_txt(txt_path)

        features = {
            'image': [],
            'width': [],
            'height': [],
            'objects': []
        }

        for image in sorted(glob.glob(f"{image_folder}/*.png")):
            img = Image.open(image)
            features['image'].append(image)
            features['width'].append(img.size[0])
            features['height'].append(img.size[1])
            features['objects'].append({
                'id': [],
                'area': [],
                'bbox': [],
                'category': []
            })

        for frame_idx, frame_info in loaded_txt.items():            
            for segmentation in frame_info:
                if segmentation.class_id in [1,2]:
                    features['objects'][frame_idx]['area'].append(rletools.area(segmentation.mask))
                    features['objects'][frame_idx]['bbox'].append(rletools.toBbox(segmentation.mask))
                    features['objects'][frame_idx]['category'].append(segmentation.class_id)
        
        self.features['image'].extend(features['image'])
        self.features['width'].extend(features['width'])
        self.features['height'].extend(features['height'])
        self.features['objects'].extend(features['objects'])
        
    def __getitem__(self, idx):
        image = Image.open(self.features['image'][idx])
        bboxes = self.features['objects'][idx]['bbox']
        categories = self.features['objects'][idx]['category']

        image = ToTensor()(image)

        return image

    def __len__(self):
        return len(self.features['image'])

if __name__ == '__main__':
    ds = KittiDataset('dataset/KITTI-MOTS', 'instances_txt', 'training', 'val.seqmap')
    print(len(ds))
    print(ds[0])
    