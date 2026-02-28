from .fasterrcnn.train import train as train_fasterrcnn
from .huggingface.train import train as train_huggingface
from .yolo.train import train as train_yolo

__all__ = [
    "train_fasterrcnn",
    "train_huggingface",
    "train_yolo"
]