import torch
from src.models import YOLOModel

from dotenv import load_dotenv
load_dotenv()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #cfg = args.cfg
    cfg = "/ghome/group02/Marina/C5-Project/Week1/configs/yolo_train.yaml"

    detector = YOLOModel(model=args.variant, device=device)
    detector.train(cfg)
