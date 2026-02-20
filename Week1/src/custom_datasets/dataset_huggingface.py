from .dataset_base import KittiDataset
from PIL import Image
from datasets import Dataset
from datasets import Image as DImage

class KittiDatasetHuggingface(KittiDataset):
    """
    Wrapper around the base KITTI dataset.

    Converts the feature dict to match what huggingface detection
    models (DeTR) expect:
        - image loaded in PIL format
    """
    def __init__(self, root: str, annotations_folder: str, image_folder: str, seqmap_file: str, transforms = None):
        super().__init__(root, annotations_folder, image_folder, seqmap_file, transforms)
        self.features['image_id'] = list(range(len(self)))
        for idx, image in enumerate(self.features['image']):
            self.features['image'][idx] = Image.open(image)
        
    
    def get_hf_ds(self):
        return Dataset.from_dict(self.features).cast_column('image', DImage())
