from .fasterrcnn.evaluation import evaluation as eval_fasterrcnn
from .huggingface.evaluation import evaluation as eval_huggingface
from .yolo.evaluation import evaluation as eval_yolo

__all__ = [
    "eval_fasterrcnn",
    "eval_huggingface",
    "eval_yolo"
]