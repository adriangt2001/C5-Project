# Week 4: Image Captioning with Transformers

## Introduction

This week focuses on **Image Captioning with Transformers** as part of the **C5 Multimodal Recognition** project. The objective is to explore modern vision-language architectures for caption generation on the **VizWiz-Captions** dataset and to study how different transformer-based modeling and adaptation strategies affect caption quality.

The work of this week includes both the evaluation of **pre-trained transformer captioning models** and the development of **fine-tuning strategies** for adapting them to the dataset. In the current repository, the implemented experiments focus on **Task 1**, where we benchmark pretrained models and fine-tune BLIP under different encoder/decoder training settings. A second stage of the work, corresponding to **Task 2**, will extend this analysis with additional experiments and comparisons.

The main approaches explored in this week are:

- **Pre-trained ViT-GPT2** for image captioning
- **Pre-trained BLIP base** and **BLIP large**
- **Selective fine-tuning of BLIP**
  - Encoder only
  - Decoder only
  - Encoder and decoder together
- **Pre-trained Llama 3.2-11B** multimodal model
- **Fine-tune LLM decoders Llama 3.2-1B** and **Llama 3.2-3B** with LoRA
- **Evaluation using BLEU-1, BLEU-2, ROUGE-L, and METEOR**
- **Qualitative comparison of generated captions**

## Dataset

We use the **VizWiz-Captions dataset**, which contains real-world images with multiple human-annotated captions.

### Data Split

We follow a custom split based on the original dataset:

- **Training set**: 90% of the original training split
- **Validation set**: 10% of the original training split
- **Test set**: original validation split, used as evaluation set for inference experiments

## Tasks Overview

The work for **Week 4** is currently organized around **Task 1**, focused on transformer-based image captioning.

- **Task 1:** Evaluate and fine-tune transformer captioning models on VizWiz-Captions
  - **Task 1.a:** Inference with pre-trained
    - **ViT-GPT2**
    - **BLIP base**
    - **BLIP large**
  - **Task 1.b:** Fine-tuning 
    - **BLIP encoder only**
    - **BLIP decoder only**
    - **BLIP encoder and decoder**
  - **Task 1.c:** Extract model metrics BLEU-1, BLEU-2, ROUGE-L, and METEOR.
  - **Task 1.d:** Comparison and discussion of the results

- **Task 2:** Evaluate and fine-tune LLMs on VizWiz-Captions
  - **Task 2.a:** Inference with pre-trained multimodal **Llama 3.2-11B**
  - **Task 2.b:** Fine-tune decoders **Llama 3.2-1B** and **Llama 3.2-3B** using LoRA.
  - **Task 2.c:** Report the models  using BLEU-1, BLEU-2, ROUGE-L, and METEOR.
  - **Task 2.d:** Compare and discuss the results


## Repository Structure

The `Week4` directory is organized as follows:

```text
Week4/
│
├── configs/
│   └── task1/                            # YAML configs for inference and finetuning experiments
├── notebooks/
│   └── qualitative_evaluation.ipynb      # Qualitative analysis of generated captions
├── scripts/
│   ├── run_inference.sh                  # Run a single inference configuration
│   ├── run_finetuning.sh                 # Run a single finetuning configuration
│   ├── run_all_inference.sh              # Run all pretrained inference experiments
│   └── run_all_inference_finetuned_models.sh
│                                         # Run inference on finetuned BLIP checkpoints
├── src/
│   ├── task1/
│   │   ├── dataset.py                    # Dataset loading, splitting, and collate functions
│   │   ├── finetuning.py                 # BLIP finetuning and validation pipeline
│   │   ├── inference.py                  # Caption generation and metric computation
│   │   ├── models.py                     # Hugging Face model and processor loading
│   │   └── __init__.py
│   ├── utils/
│   │   ├── io.py                         # YAML loading utilities
│   │   └── metrics.py                    # BLEU, ROUGE-L, and METEOR computation
│   └── main.py                           # Main entry point for Week 4
└── README.md                             # README for Week 4
```

## Installation

To run the code in this repo you must first install all needed libraries using conda with the help of the `environment.yml` file. This code is tested under python version 3.12.

```bash
cd Week4
conda env create -f environment.yml
conda activate c5
```

## General Usage

All Week 4 experiments are launched through the same entry point using a task-specific YAML configuration file.

```bash
cd Week4
python -m src.main inference --config configs/task1/blip-base-pretrained.yaml
```

For finetuning:

```bash
cd Week4
python -m src.main finetuning --config configs/task1/blip-finetuning-encoder.yaml
```

## Available Configurations

### Pre-trained inference

- `configs/task1/vit-gpt2-pretrained.yaml`
- `configs/task1/blip-base-pretrained.yaml`
- `configs/task1/blip-large-pretrained.yaml`

### BLIP fine-tuning

- `configs/task1/blip-finetuning-encoder.yaml`
- `configs/task1/blip-finetuning-decoder.yaml`
- `configs/task1/blip-finetuning-encoder-decoder.yaml`
- `configs/task1/blip-finetuning-full.yaml`

### Inference on fine-tuned checkpoints

- `configs/task1/blip-inference-encoder.yaml`
- `configs/task1/blip-inference-decoder.yaml`
- `configs/task1/blip-inference-full.yaml`

## Running the Provided Scripts

To run a single pretrained inference experiment:

```bash
cd Week4
bash scripts/run_inference.sh configs/task1/blip-base-pretrained.yaml
```

To run a finetuning experiment:

```bash
cd Week4
bash scripts/run_finetuning.sh configs/task1/blip-finetuning-encoder.yaml
```

To evaluate all pre-trained captioning models:

```bash
cd Week4
bash scripts/run_all_inference.sh
```

To evaluate all saved fine-tuned BLIP models:

```bash
cd Week4
bash scripts/run_all_inference_finetuned_models.sh
```

## Outputs

The pipeline produces:

- **JSON result files** with generated captions and evaluation metrics in `results/task1/`
- **Fine-tuned BLIP checkpoints** in the directory specified by `best_trained_model_path`
- **W&B logs** for training and evaluation experiments
- **Qualitative analysis notebook outputs** in `notebooks/qualitative_evaluation.ipynb`

## Notes

- Validation during finetuning is performed on a held-out split created from the original training annotations.
- Baseline validation metrics are computed at **epoch 0** before training starts.
- The best fine-tuned model is selected according to the **highest METEOR score** on the validation split.
