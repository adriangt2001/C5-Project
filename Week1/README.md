# Week 1: Object Detection

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
│   ├── task_c/          # Inference with pre-trained models
│   ├── task_d/          # Evaluation using COCO metrics
│   ├── task_e/          # Fine-tuning 
│   ├── task_f/          # Fine-tuning in a different domain
│   ├── task_g/          # Comparative analysis
│   └── task_h/          # Fine-tuning RT-DETR
└── README.md            # README for Week 1
```

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

Results are saved in the ```Week1/results/task_c``` folder.

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

## Task f

## Task g

## Task h