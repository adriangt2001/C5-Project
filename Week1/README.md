# Week 1: Object Detection

[View Presentation](https://www.canva.com/design/DAHBmM5MBbU/E24ryIqKLlVz7kRqTmjn-Q/edit?utm_content=DAHBmM5MBbU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Introduction

This week focuses on the Object Detection task as part of the C5 Multimodal Recognition project. The objective is to explore, evaluate, and compare different state-of-the-art object detection models using PyTorch-based frameworks.

We work with the KITTI-MOTS dataset and implement three main detection approaches:

- Faster R-CNN (Torchvision)

- DETR (Hugging Face)

- YOLOv(>8) (Ultralytics)

## Dataset

We use the KITTI-MOTS dataset, which contains instance-level annotations for cars and pedestrians in urban driving scenarios.

### Classes
- Car (class id = 1)
- Pedestrian (class id = 2)

### Annotation Format

Annotations are provided as instance-level segmentation masks encoded either as PNG images or in TXT format using COCO-style run-length encoding (RLE).

### Data Split

We follow the official training and validation partitions.

- **Training set:** 12 sequences  
  - 8,073 pedestrian masks  
  - 18,831 car masks  

- **Validation set:** 9 sequences  
  - 3,347 pedestrian masks  
  - 8,068 car masks  

## Tasks Overview

The work for Week 1 is structured according to the project tasks:
 
- **Task (c):** Inference with pre-trained Faster R-CNN, DETR and YOLO on KITTI-MOTS  
- **Task (d):** Evaluation using COCO metrics  
- **Task (e):** Fine-tuning on KITTI-MOTS   
- **Task (f):** Fine-tuning under domain shift  
- **Task (g):** Comparative analysis of detection models  
- **Task (h):** Fine-tuning RT-DETR and extended analysis 

Tasks (a) and (b) corresponded to environment setup and dataset familiarization and are not explicitly included in this repository.

## Repository Structure

The Week1 directory is organized as follows:

```
Week1/
│
├── src/
│   ├── custom_datasets/ # Dataset loading and preprocessing
│   ├── models/          # Model wrappers and utilities
│   ├── notebooks/       # Notebooks
│   ├── task_c/          # Inference with pre-trained models
│   ├── task_d/          # Evaluation using COCO metrics
│   ├── task_e/          # Fine-tuning 
│   ├── task_f/          # Fine-tuning in a different domain
│   └── task_h/          # Fine-tuning RT-DETR
├── environment.yml      # Environment yaml
└── README.md            # README for Week 1
```

> **Note:** Task **g** has no dedicated folder, as it focuses on benchmarking and comparing the results obtained across the previous tasks.

## Installation
To run the code in this repo you must first install all needed libraries using conda with the help of the ```environment.yml``` file. This code is tested under python version 3.12.

```bash
cd Week1
conda env create -f environment.yml
conda activate c5
```

## Task c
In Task C, we run inference of different models to get some qualitative results. Those models are Faster R-CNN, DeTR and YOLO. To run the code and get some results run:

```bash
# To run FasterRCNN
python -m src.task_c.main_inference --model fasterrcnn --variant resnet50_fpn_v2 --batch_size 16

# To run DeTR
python -m src.task_c.main_inference --model detr --variant facebook/detr-resnet-50 --batch_size 16

# To run YOLO
python -m src.task_c.main_inference --model yolo --variant yolov10m.pt --batch_size 16
```

For more information of the different arguments run:

```bash
python -m src.task_c.main_inference --help
```

Results are logged in **wandb** by default under the project **C5Week1**.

**Tested variants model variants**
* Faster R-CNN
  - resnet50_fpn_v2
  - resnet50_fpn
  - mobilenet_v3_large_320_fpn
  - mobilenet_v3_large_fpn
  - 
* DETR:
  - facebook/detr-resnet-50
  - microsoft/conditional-detr-resnet-50
* YOLO:
  - yolov10m.pt

## Task d
In Task D, we evaluate the different models and their variants to get quantitative results using the COCO metrics. To run the code and get some results run:

```bash
# To run FasterRCNN
python -m src.task_d.main_evaluation --model fasterrcnn --variant resnet50_fpn_v2 --batch_size 8

# To run DeTR
python -m src.task_d.main_evaluation --model detr --variant facebook/detr-resnet-50 --batch_size 8

# To run YOLO
python -m src.task_d.main_evaluation --model yolo --variant yolov10m.pt --batch_size 8
```

For more information of the different arguments run:

```bash
python -m src.task_d.main_evaluation --help
```

Results are logged in **wandb** by default under the project **C5Week1**.

**Tested variants model variants**
* Faster R-CNN
  - resnet50_fpn_v2
  - resnet50_fpn
  - mobilenet_v3_large_320_fpn
  - mobilenet_v3_large_fpn
  - 
* DETR:
  - facebook/detr-resnet-50
  - microsoft/conditional-detr-resnet-50
* YOLO:
  - yolov10m.pt

## Task e
In Task E, we fine-tune pre-trained detectors on KITTI-MOTS (similar domain) to measure the effect of transfer learning. We support Faster R-CNN (Torchvision), DeTR (HuggingFace) and YOLO (Ultralytics). Only one version of each in this case.

```bash
# To fine-tune Faster R-CNN
python -m src.task_e.main_training \
  --model fasterrcnn \
  --variant resnet50_fpn \
  --dataset dataset/KITTI-MOTS \
  --batch_size 32 \
  --lr 5e-5 \
  --epochs 50

# To fine-tune DeTR (HuggingFace)
python -m src.task_e.main_training \
  --model detr \
  --variant facebook/detr-resnet-50 \
  --dataset dataset/KITTI-MOTS \
  --batch_size 32 \
  --lr 5e-5 \
  --epochs 50

# To fine-tune YOLO (Ultralytics)
python -m src.task_e.main_training \
  --model yolo \
  --variant yolov10m.pt \
  --dataset dataset/KITTI-MOTS \
  --batch_size 32 \
  --lr 5e-5 \
  --epochs 50
```

For more information of the different arguments run:

```bash
python -m src.task_e.main_train --help
```

Results are logged in **wandb** by default under the project **C5Week1**.



## Task f

## Task h