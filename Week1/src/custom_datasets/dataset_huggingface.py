from .dataset_base import KittiDataset
from PIL import Image

class KittiDatasetHuggingface(KittiDataset):
    """
    Wrapper around the base KITTI dataset.

    Converts the feature dict to match what huggingface detection
    models (DeTR) expect:
        - image loaded in PIL format
    """
    def __init__(self, root: str, annotations_folder: str, image_folder: str, seqmap_file: str, transforms = None):
        super().__init__(root, annotations_folder, image_folder, seqmap_file, transforms)
        
        for idx, image in enumerate(self.features['image']):
            self.features['image'][idx] = Image.open(image)        
