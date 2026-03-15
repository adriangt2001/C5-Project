# Week 2: Object Segmentation

[View Presentation](https://www.canva.com/design/DAHDYgXHhck/kKZ6azgZDvShzLwuqEQq4g/edit?utm_content=DAHDYgXHhck&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Introduction
This week focuses on the **Object Segmentation** task as part of the **C5 Multimodal Recognition** project. The objective is to explore, evaluate, fine-tune, and compare different segmentation strategies based on the **Segment Anything Model (SAM)** and related prompt-based approaches.

We work with the **KITTI-MOTS** dataset and study how segmentation performance changes depending on the prompt type and the source of those prompts.

The main approaches explored in this week are:

- **SAM with point prompts**
- **Grounded SAM with text prompts**
- **SAM with bounding-box prompts** obtained from the best object detector from Week 1
- **Fine-tuned SAM** for instance segmentation
- **Evaluation under domain shift**
- **Comparative analysis across prompts and models**

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


The work for **Week 2** is structured according to the project tasks:

- **Task (a):** Inference and evaluation of pre-trained SAM on KITTI-MOTS using **point prompts**
- **Task (b):** Inference and evaluation of pre-trained SAM on KITTI-MOTS using **text prompts** with **Grounded SAM**
- **Task (c):** Inference with **bounding-box prompts** obtained from the best detection model from Week 1
- **Task (d):** Comparison between **Grounded SAM** and **bbox-prompted SAM**
- **Task (e):** Fine-tuning the **prompt decoder of SAM** for instance segmentation on KITTI-MOTS
- **Task (f):** Inference under **domain shift** using pre-trained and fine-tuned SAM
- **Task (g):** Comparative analysis of the different prompt types
- **Task (h):** *(Optional)* Semantic segmentation on KITTI-MOTS

## Repository Structure

The Week2 directory is organized as follows:

```
Week2/
│
├── configs/             # YAML configuration files for each task
├── src/
│   ├── utils/           # Common utility functions
│   ├── task_a/          # Code for task a
│   ├── task_b/          # code for task b
│   ├── task_c/          # Code for task c
│   ├── task_e/          # Code for task e
│   ├── task_f/          # Code for task f
│   ├── task_g/          # Code for task g
│   ├── task_h/          # Code for task h
│   └── main.py          # Entry for Week2
├── environment.yml      # Environment yaml
└── README.md            # README for Week 2
```

> **Note:** **Task (d)** does not have a dedicated folder in the repository, since it focuses on comparing the results obtained in **Task (b)** and **Task (c)** rather than running a separate pipeline.

## Installation
To run the code in this repo you must first install all needed libraries using conda with the help of the ```environment.yml``` file. This code is tested under python version 3.12.

```bash
cd Week2
conda env create -f environment.yml
conda activate c5
```

## General Usage
All experiments in Week 2 are launched through the same entry point using a task-specific YAML configuration file.

```bash
cd C5-project/Week2
python -m src.main --config configs/task_X.yaml
```

Replace `task_X.yaml` with the corresponding configuration file for the task you want to run.