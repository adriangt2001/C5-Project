from PIL import Image
from .dataset_base import KittiDataset


class KittiDatasetUltralytics(KittiDataset):
    """
    Wrapper around the base KITTI dataset
    
    Returns the output image as a PIL which is more suitable for 
    Ultralytics expected input
    """
    def __getitem__(self, idx):
        # Load image
        img_path = self.features["image"][idx]
        image = Image.open(img_path).convert("RGB")

        return image