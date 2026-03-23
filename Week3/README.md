# Week 3: Image Captioning

[View Presentation](https://canva.link/9uoearg1nfqn0hm)

## Introduction
This week focuses on the **Image Captioning** task as part of the **C5 Multimodal Recognition** project. The objective is to generate descriptive sentences from images by combining computer vision and natural language processing.

We implement and evaluate different encoder-decoder architectures, analyzing how architectural choices and training strategies affect caption quality.

The main approaches explored in this week are:

- Baseline CNN-RNN model (ResNet18 + GRU)
- Different encoder backbones (ResNet18 vs. ResNet50)
- Different decoders (GRU vs. LSTM)
- Different tokenization levels (character-level vs. subword-level and word-level)
- Attention mechanism
- Teacher forcing vs. Scheduled sampling strategies
- Comparative analysis across experiments

## Dataset

We use the **VizWiz-Captions dataset**, which contains real-world images with multiple human-annotated captions.

### Data Split

We follow a custom split based on the original dataset:

- **Training set**: 90% of the original training split
- **Validation set**: 10% of the original training split
- **Test set**: original validation split (used as test set)

## Tasks Overview

The work for **Week 3** is structured according to the project tasks:

- **Task 1:** Implement a baseline encoder-decoder model (ResNet18 + GRU)
    - **Task 1.a:** Change the encoder choosing ONE alternative encoder module: ResNet50
    - **Task 1.b** Change the decoder choosing ONE alternative decoder module: LSTM
    - **Task 1.c**: Change the text representation level from character to subword and word
    - **Task 1.d:** Implement an attention mechanism
- **Task 2:** Train and evaluate your models. Compare among them and with the baseline

## Repository Structure

The Week3 directory is organized as follows:

```
Week3/
│
├── captioning.py                                     # Model and captioning logic 
├── dataset.py                                        # Data loading and preprocessing 
├── main.py                                           # Training and evaluation entry point 
├── Baseline Model and Metrics.ipynb                  # Baseline code
├── qualitative_evaluation.ipynb                      # Qualitative analysis of results
│  
├── initial_experiments_sampling.sh                   # Initial sampling experiments 
├── run_captioning_experiments_attention.sh           # Attention experiments 
├── run_captioning_experiments_frozen_encoders.sh     # Frozen encoder experiments 
├── run_captioning_experiments_scheduled_sampling.sh  # Scheduled Sampling experiments 
│ 
├── environment.yml                                   # Environment for Week 3
├── README.md                                         # README for Week 3
```

## Installation
To run the code in this repo you must first install all needed libraries using conda with the help of the ```environment.yml``` file. This code is tested under python version 3.12.

```bash
cd Week3
conda env create -f environment.yml
conda activate c5
```

## General Usage
All experiments in Week 3 are executed through bash scripts, which define different configurations and training setups.

To run an experiment, execute one of the provided scripts:
```bash
cd Week3
bash initial_experiments_sampling.sh
```
Other available scripts:
```bash
bash run_captioning_experiments_attention.sh
bash run_captioning_experiments_frozen_encoders.sh
bash run_captioning_experiments_scheduled_sampling.sh
```
These scripts handle:

- Training the model with specific configurations
- Evaluating the best checkpoint
- Saving results and metrics automatically
