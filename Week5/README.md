# Week5 - SDXL Inference and Analysis

This folder contains Week5 code for inference sweeps and model comparisons.
All commands below assume you are in the `Week5` directory.

## Run Task A (model comparison)

```bash
cd C5-Project/Week5
uv run python -m src.main task_a --config configs/task_a/compare_open_models.yaml
```

This command compares the models listed in `configs/task_a/compare_open_models.yaml` and saves outputs under `outputs/task_a`.

## Compare input modes for Task A

```bash
cd C5-Project/Week5
uv run python src/task_a/compare_inputs.py --config configs/task_a/compare_inputs.yaml
```

This command runs the input comparison script using the config in `configs/task_a/compare_inputs.yaml`.

## Run Task B (SDXL Turbo inference sweep)

```bash
cd C5-Project/Week5
uv run python -m src.main task_b --config configs/task_b/sdxl_turbo_inference_sweep.yaml
```

This command runs the sweep defined in `configs/task_b/sdxl_turbo_inference_sweep.yaml` and saves outputs under `outputs/task_b`.

## Generate analysis plots for Task B

```bash
cd C5-Project/Week5
uv run python src/task_b/compare_parameters.py
```

This script reads `outputs/task_b/summary.json` and saves comparison plots to `outputs/task_b/analysis`.

## Run Task D (standalone prompt to image generation)

```bash
cd C5-Project/Week5
uv run python src/task_d/generate_images.py --config configs/task_d/sdxl_turbo_generate.yaml
```

This script reads prompts from `configs/task_d/sdxl_turbo_generate.yaml` and saves the generated images under `outputs/task_d`.
