# Week5 - Caption Analysis, Synthetic Generation, and Finetuning

This folder contains the Week 5 code for:

- Task C: error analysis and evaluation of the captioning model
- Task D: caption sampling, FLUX image generation, and synthetic annotation building
- Task E: finetuning and inference for the captioning model

All commands below assume you are in the `Week5` directory.

## Run Task A (model comparison)

```bash
cd C5-Project/Week5
uv run python -m src.main task_a --config configs/task_a/compare_open_models.yaml
```

This compares the models listed in `configs/task_a/compare_open_models.yaml` and saves outputs under `outputs/task_a`.

## Compare input modes for Task A

```bash
cd C5-Project/Week5
uv run python src/task_a/compare_inputs.py --config configs/task_a/compare_inputs.yaml
```

This runs the input comparison script using the config in `configs/task_a/compare_inputs.yaml`.

## Run Task B (SDXL Turbo inference sweep)

```bash
cd C5-Project/Week5
uv run python -m src.main task_b --config configs/task_b/sdxl_turbo_inference_sweep.yaml
```

This runs the sweep defined in `configs/task_b/sdxl_turbo_inference_sweep.yaml` and saves outputs under `outputs/task_b`.

## Generate analysis plots for Task B

```bash
cd C5-Project/Week5
uv run python src/task_b/compare_parameters.py
```

This script reads `outputs/task_b/summary.json` and saves comparison plots to `outputs/task_b/analysis`.

## Task C - Analyze Captioning Failures

The main executable components for Task C are:

- `src/task_c/find_problems_current_models.ipynb`
- `src/task_c/evaluate_finetuned_checkpoint.py`

### Inspect model outputs in the notebook

Open and run:

- `src/task_c/find_problems_current_models.ipynb`

This notebook is used to inspect the BLIP predictions stored in:

- `src/task_c/blip-finetuned-full.json`

The notebook includes analyses for:

- fallback caption frequency
- placeholder-caption issues in the annotations
- disjoint theme classification
- per-theme error patterns
- train/validation caption filtering and statistics

### Evaluate a finetuned checkpoint by theme

Script:

- `src/task_c/evaluate_finetuned_checkpoint.py`

```bash
cd C5-Project/Week5
uv run python src/task_c/evaluate_finetuned_checkpoint.py \
  --config configs/task_e/blip-finetuning-sd-data.yaml \
  --checkpoint finetuned_models/sd_data \
  --output-json outputs/task_c/sd_data_theme_eval.json \
  --val-annotations data/annotations/val.json
```

This evaluates a finetuned BLIP checkpoint on the validation set and reports:

- overall BLEU-1, BLEU-2, ROUGE-L, and METEOR
- per-theme metrics using the disjoint theme assignment

## Task D - Generate Synthetic Images from Captions

The main executable components for Task D are:

- `src/task_d/sample_random_captions.py`
- `src/task_d/split_captions_shards.py`
- `src/main.py task_d_flux`
- `src/main.py build_synthetic_annotations`

Task D is organized in four steps:

1. Sample captions from the filtered train annotations.
2. Split the sampled captions into shards.
3. Generate synthetic images with FLUX.
4. Build a COCO-style synthetic annotation file for finetuning.

### 1. Sample captions by theme

Script:

- `src/task_d/sample_random_captions.py`

```bash
cd C5-Project/Week5
uv run python src/task_d/sample_random_captions.py \
  --train_annotations_path data/annotations/train_filtered.json \
  --output_captions_path data/generated/captions.json \
  --samples_per_weight 100 \
  --seed 42 \
  --drop_fallback \
  --unique_captions
```

This script:

- assigns each image to a theme using keyword-based heuristics over its references
- samples captions per theme according to predefined theme weights
- stores the result in `data/generated/captions.json`

The output JSON includes:

- `captions_by_theme`
- `theme_summary`

### 2. Split captions into shards

Script:

- `src/task_d/split_captions_shards.py`

```bash
cd C5-Project/Week5
uv run python src/task_d/split_captions_shards.py \
  --input_json data/generated/captions.json \
  --num_shards 3 \
  --output_dir data/generated/shards
```

This creates balanced shard files such as:

- `data/generated/shards/captions_shard_1_of_3.json`
- `data/generated/shards/captions_shard_2_of_3.json`
- `data/generated/shards/captions_shard_3_of_3.json`

Captions belonging to the same source image are kept in the same shard.

### 3. Generate images with FLUX

Script:

- `src/main.py` with subcommand `task_d_flux`

The FLUX generation entrypoint is exposed through `src/main.py` as `task_d_flux`.

Example for shard 1:

```bash
cd C5-Project/Week5
uv run python -m src.main task_d_flux --config configs/task_d/flux_shard_1.json
```

Equivalent commands for the other shards:

```bash
cd C5-Project/Week5
uv run python -m src.main task_d_flux --config configs/task_d/flux_shard_2.json
uv run python -m src.main task_d_flux --config configs/task_d/flux_shard_3.json
```

The provided config files are:

- `configs/task_d/flux_shard_1.json`
- `configs/task_d/flux_shard_2.json`
- `configs/task_d/flux_shard_3.json`

These configs run `diffusers/FLUX.2-dev-bnb-4bit` using:

- prompts loaded from a shard JSON
- prompt prefix `"low quality picture of a " + caption`
- separate output directories under `results/`

Each run writes:

- generated images in `results/flux_run_shard_X/images/`
- an incremental `generation_report.jsonl`

The generation order is balanced across themes, so if execution stops early the partial output is still more evenly distributed.

### 4. Build synthetic annotations from generated images

Script:

- `src/main.py` with subcommand `build_synthetic_annotations`

After generation finishes, build the synthetic annotation file with:

```bash
cd C5-Project/Week5
uv run python -m src.main build_synthetic_annotations \
  --config configs/task_d/build_synthetic_annotations_flux.json
```

This reads the FLUX `generation_report.jsonl` files and creates:

- `data/annotations/train_synthetic_flux.json`
- symbolic links to the generated images under `data/synthetic_flux/`

The generated JSON follows the same COCO-style structure used by the training annotations:

- `info`
- `images`
- `annotations`

### Build synthetic annotations from a Stable Diffusion manifest

If you already have synthetic images generated by another model and a manifest such as `captions_with_images.json`, run:

```bash
cd C5-Project/Week5
uv run python -m src.main build_synthetic_annotations \
  --config configs/task_d/build_synthetic_annotations_sd.json
```

This creates:

- `data/annotations/train_synthetic_sd.json`
- symbolic links under `data/synthetic_sd/`

Image filenames are preserved, and the builder creates symbolic links instead of copying the images.

## Task E - Finetune and Evaluate Captioning Models

The main executable components for Task E are:

- `src/main.py` with subcommand `finetuning`
- `src/main.py` with subcommand `finetuning_sd_data`
- `src/main.py` with subcommand `inference`
- `src/task_c/evaluate_finetuned_checkpoint.py`

Task E covers two slightly different finetuning paths:

- `finetuning`: uses the standard dataset loader and can combine original data with synthetic annotation files already integrated into `data/annotations/`
- `finetuning_sd_data`: uses the synthetic-directory loader and can read one or more external synthetic folders through `synthetic_data_dirs`

### Finetune with original data only or with integrated synthetic annotations

Script:

- `src/main.py` with subcommand `finetuning`

This is the right entrypoint when your synthetic data has already been converted into annotation files under `data/annotations/`.

In the current code, this training path reads synthetic annotations from the filenames expected by `src/task_e/finetuning.py`, for example:

- `data/annotations/train_synthetic.json`
- `data/annotations/train_synthetic_sd.json`

Example config files:

- `configs/task_e/blip-finetuning-no-aug.yaml`
- `configs/task_e/blip-finetuning-sd-flux.yaml`

Run without augmentation:

```bash
cd C5-Project/Week5
uv run python -m src.main finetuning --config configs/task_e/blip-finetuning-no-aug.yaml
```

Run with both SD and FLUX synthetic annotations:

```bash
cd C5-Project/Week5
uv run python -m src.main finetuning --config configs/task_e/blip-finetuning-sd-flux.yaml
```

What this script does:

- loads `data/annotations/train_filtered.json`
- splits the original training set into train and validation subsets
- optionally adds synthetic samples from annotation files
- logs how many original and synthetic samples are loaded
- trains the BLIP captioning model
- evaluates after every epoch
- saves the best checkpoint according to validation METEOR

The best checkpoint path is controlled by:

- `best_trained_model_path`

Note:

- if you build new synthetic annotation files with different names, make sure they match the filenames expected by `src/task_e/finetuning.py` before using this entrypoint

### Finetune using one or more synthetic generation folders

Script:

- `src/main.py` with subcommand `finetuning_sd_data`

Use this version when you want to train from original VizWiz data plus one or more synthetic folders that still contain their own `captions_with_images.json` manifest.

Example config files:

- `configs/task_e/blip-finetuning-sd-data.yaml`
- `configs/task_e/blip-finetuning-sd-data-extended.yaml`
- `configs/task_e/blip-finetuning-flux.yaml`

Run with one synthetic directory:

```bash
cd C5-Project/Week5
uv run python -m src.main finetuning_sd_data --config configs/task_e/blip-finetuning-sd-data.yaml
```

Run with multiple synthetic directories:

```bash
cd C5-Project/Week5
uv run python -m src.main finetuning_sd_data --config configs/task_e/blip-finetuning-sd-data-extended.yaml
```

This script:

- loads the original filtered VizWiz train set
- loads synthetic samples from the directories listed in `synthetic_data_dirs`
- ignores missing image files instead of crashing
- logs how many samples come from each source
- trains and validates the model
- saves the best checkpoint by METEOR

### Run inference with a finetuned model

Script:

- `src/main.py` with subcommand `inference`

Example config files:

- `configs/task_e/blip-inference-no-aug.yaml`
- `configs/task_e/blip-inference-flux.yaml`

Example command:

```bash
cd C5-Project/Week5
uv run python -m src.main inference --config configs/task_e/blip-inference-flux.yaml
```

This script:

- loads a finetuned checkpoint from `model_name`
- runs caption generation on `data/annotations/val.json`
- computes BLEU-1, BLEU-2, ROUGE-L, and METEOR
- measures total runtime and throughput
- saves predictions and metrics to the configured `output_file`

### Evaluate a finetuned checkpoint with per-theme metrics

Script:

- `src/task_c/evaluate_finetuned_checkpoint.py`

This script is especially useful after training because it complements the standard inference output with per-theme results.

Example command:

```bash
cd C5-Project/Week5
uv run python src/task_c/evaluate_finetuned_checkpoint.py \
  --config configs/task_e/blip-finetuning-sd-data.yaml \
  --checkpoint finetuned_models/sd_data \
  --output-json outputs/task_c/sd_data_theme_eval.json \
  --val-annotations data/annotations/val.json
```

It reports:

- overall validation metrics
- per-theme captioning metrics
- a JSON file with predictions, references, and aggregated scores
