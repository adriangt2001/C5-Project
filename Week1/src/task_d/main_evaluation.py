import argparse
from src.task_d.fasterrcnn.evaluation import evaluation as eval_fasterrcnn

def main_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fasterrcnn")
    parser.add_argument("--variant", type=str, default="resnet50_fpn_v2")
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()

    if args.model == "fasterrcnn":
        eval_fasterrcnn(args)
        
    elif args.model == "detr":
        pass
        
    elif args.model == "yolo":
        pass

if __name__ == "__main__":
    main_evaluation()