# C5-Project: Multimodal Recognition (Team 2)

## Members
- Lore Oregi Lauzirika
- Júlia Garcia Torné
- Adrián García Tapia
- Marina Rosell Murillo

## Project Structure

This repository is structured by weeks. Each directory contains its own source code, experiments, and a dedicated README with detailed technical documentation.

```
├── .gitignore
├── README.md              # Global overview 
└── WeekX/                 # Week X
    ├── README.md          # Technical details for Week X
    └── src/               # Code for Week X
```

## Week 1: Object Detection

In Week 1 of the C5 Project (Multimodal Recognition), we focus on the Object Detection task using different frameworks. The objective is to explore and compare different state-of-the-art detection models, including Faster R-CNN (Torchvision), DETR (Hugging Face), and YOLO (Ultralytics), following the project guidelines. 

We perform inference, evaluation using COCO metrics, and fine-tuning on the KITTI-MOTS dataset. The goal is to analyze performance differences across models in terms of accuracy, robustness, inference time, and generalization capabilities.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 1 README](./Week1/README.md).

# Week 2: Object Segmentation

In Week 2 of the C5 Project, we focus on the Object Segmentation task using the Segment Anything Model (SAM). The objective is to explore prompt-based segmentation approaches and evaluate the performance of pre-trained and fine-tuned models on the KITTI-MOTS dataset.

We perform inference using different types of prompts (points, text, and bounding boxes), including bounding boxes obtained from the object detection models developed in Week 1. Additionally, we compare results across different prompting strategies and evaluate the impact of fine-tuning the SAM prompt decoder for instance segmentation.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 2 README](./Week2/README.md).


