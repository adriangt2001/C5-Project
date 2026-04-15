#!/usr/bin/env python3
"""
Generate informative parameter comparison plots for diffusion experiments.

Designed for analysis of:
- DDPM vs DDIM
- positive-only vs positive+negative prompting
- CFG strength
- number of denoising steps
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from PIL import Image


Experiment = Dict[str, Any]


def load_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "experiments" not in data or not isinstance(data["experiments"], list):
        raise ValueError("summary.json must contain a list under 'experiments'")

    return data


def normalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    return value


def matches(exp: Mapping[str, Any], criteria: Mapping[str, Any]) -> bool:
    return all(normalize(exp.get(k)) == normalize(v) for k, v in criteria.items())


def find_experiment(
    experiments: Sequence[Experiment],
    criteria: Mapping[str, Any],
) -> Optional[Experiment]:
    matches_found = [exp for exp in experiments if matches(exp, criteria)]
    if not matches_found:
        return None
    return matches_found[0]


def load_image(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def draw_panel(ax, exp: Optional[Experiment], criteria: Mapping[str, Any], annotate: bool = True) -> None:
    if exp is None:
        ax.text(
            0.5, 0.5, "No experiment found",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=10
        )
        ax.axis("off")
        return

    img = load_image(Path(exp["image_path"]))
    if img is None:
        ax.text(
            0.5, 0.5, "Image missing/unreadable",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=10
        )
        ax.axis("off")
        return

    ax.imshow(img)
    ax.axis("off")

    if annotate:
        annotation = (
            f"scheduler={exp.get('scheduler', '?')}\n"
            f"steps={exp.get('num_inference_steps', '?')}, "
            f"cfg={exp.get('guidance_scale', '?')}\n"
            f"prompt={exp.get('prompt_mode', '?')}"
        )
 

def make_title_from_fixed(fixed_params: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in fixed_params.items())


def create_matrix_plot(
    experiments: Sequence[Experiment],
    row_param: str,
    col_param: str,
    row_values: Sequence[Any],
    col_values: Sequence[Any],
    fixed_params: Mapping[str, Any],
    output_path: Path,
    title: str,
) -> None:
    """Create a 2D grid plot with rows and columns representing parameters."""
    n_rows = len(row_values)
    n_cols = len(col_values)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 4.8 * n_rows),
        squeeze=False
    )

    for r, row_val in enumerate(row_values):
        for c, col_val in enumerate(col_values):
            ax = axes[r][c]
            criteria = dict(fixed_params)
            criteria[row_param] = row_val
            criteria[col_param] = col_val

            exp = find_experiment(experiments, criteria)
            draw_panel(ax, exp, criteria, annotate=True)

            if r == 0:
                ax.set_title(f"{col_param} = {col_val}", fontsize=12, fontweight="bold", pad=10)
            if c == 0:
                ax.text(
                    -0.08, 0.5,
                    f"{row_param} = {row_val}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )

    fixed_str = make_title_from_fixed(fixed_params)
    fig.suptitle(
        f"{title}\nFixed: {fixed_str}",
        fontsize=16,
        fontweight="bold",
    )

    fig.text(
        0.5, 0.01,
        f"Rows = {row_param} | Columns = {col_param}",
        ha="center",
        fontsize=11
    )

    fig.tight_layout(rect=(0.03, 0.04, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {output_path}")


def create_line_sweep_plot(
    experiments: Sequence[Experiment],
    varying_param: str,
    varying_values: Sequence[Any],
    fixed_params: Mapping[str, Any],
    output_path: Path,
    title: str,
) -> None:
    """Create a 1-row sweep plot for a single varying parameter."""
    n_cols = len(varying_values)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 5), squeeze=False)
    axes = axes[0]

    for i, value in enumerate(varying_values):
        criteria = dict(fixed_params)
        criteria[varying_param] = value
        exp = find_experiment(experiments, criteria)

        draw_panel(axes[i], exp, criteria, annotate=True)
        axes[i].set_title(
            f"{varying_param} = {value}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    fixed_str = make_title_from_fixed(fixed_params)
    fig.suptitle(
        f"{title}\nFixed: {fixed_str}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.90))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {output_path}")


def generate_report_plots(experiments: Sequence[Experiment], output_dir: Path) -> None:
    """
    Generate assignment-oriented plots for:
    - DDPM vs DDIM
    - positive-only vs positive+negative prompting
    - CFG strength
    - denoising steps
    """

    # 1) Scheduler vs Steps at fixed prompt mode and CFG
    create_matrix_plot(
        experiments=experiments,
        row_param="scheduler",
        col_param="num_inference_steps",
        row_values=["ddpm", "ddim"],
        col_values=[1, 4, 10],
        fixed_params={
            "guidance_scale": 0.0,
            "prompt_mode": "positive_and_negative",
        },
        output_path=output_dir / "scheduler_vs_steps_cfg0_posneg.png",
        title="Effect of Scheduler and Denoising Steps",
    )

    # 2) Scheduler vs Prompt mode at fixed steps and CFG
    create_matrix_plot(
        experiments=experiments,
        row_param="scheduler",
        col_param="prompt_mode",
        row_values=["ddpm", "ddim"],
        col_values=["positive_only", "positive_and_negative"],
        fixed_params={
            "guidance_scale": 0.0,
            "num_inference_steps": 4,
        },
        output_path=output_dir / "scheduler_vs_promptmode_steps4_cfg0.png",
        title="Effect of Scheduler and Prompt Mode",
    )

    # 3) CFG sweep for DDPM
    create_line_sweep_plot(
        experiments=experiments,
        varying_param="guidance_scale",
        varying_values=[0.0, 0.5, 1.0, 1.5, 2.0],
        fixed_params={
            "scheduler": "ddpm",
            "num_inference_steps": 4,
            "prompt_mode": "positive_and_negative",
        },
        output_path=output_dir / "cfg_sweep_ddpm_steps4_posneg.png",
        title="Effect of CFG Strength (DDPM)",
    )

    # 4) CFG sweep for DDIM
    create_line_sweep_plot(
        experiments=experiments,
        varying_param="guidance_scale",
        varying_values=[0.0, 0.5, 1.0, 1.5, 2.0],
        fixed_params={
            "scheduler": "ddim",
            "num_inference_steps": 4,
            "prompt_mode": "positive_and_negative",
        },
        output_path=output_dir / "cfg_sweep_ddim_steps4_posneg.png",
        title="Effect of CFG Strength (DDIM)",
    )

    # 5) Step sweep for DDPM
    create_line_sweep_plot(
        experiments=experiments,
        varying_param="num_inference_steps",
        varying_values=[1, 4, 8, 10],
        fixed_params={
            "scheduler": "ddpm",
            "guidance_scale": 0.0,
            "prompt_mode": "positive_and_negative",
        },
        output_path=output_dir / "steps_sweep_ddpm_cfg0_posneg.png",
        title="Effect of Number of Denoising Steps (DDPM)",
    )

    # 6) Step sweep for DDIM
    create_line_sweep_plot(
        experiments=experiments,
        varying_param="num_inference_steps",
        varying_values=[1, 4, 8, 10],
        fixed_params={
            "scheduler": "ddim",
            "guidance_scale": 0.0,
            "prompt_mode": "positive_and_negative",
        },
        output_path=output_dir / "steps_sweep_ddim_cfg0_posneg.png",
        title="Effect of Number of Denoising Steps (DDIM)",
    )

    # 7) Prompt mode comparison under DDPM
    create_line_sweep_plot(
        experiments=experiments,
        varying_param="prompt_mode",
        varying_values=["positive_only", "positive_and_negative"],
        fixed_params={
            "scheduler": "ddpm",
            "guidance_scale": 0.0,
            "num_inference_steps": 4,
        },
        output_path=output_dir / "promptmode_ddpm_steps4_cfg0.png",
        title="Effect of Positive-Only vs Positive+Negative Prompting (DDPM)",
    )

    # 8) Prompt mode comparison under DDIM
    create_line_sweep_plot(
        experiments=experiments,
        varying_param="prompt_mode",
        varying_values=["positive_only", "positive_and_negative"],
        fixed_params={
            "scheduler": "ddim",
            "guidance_scale": 0.0,
            "num_inference_steps": 4,
        },
        output_path=output_dir / "promptmode_ddim_steps4_cfg0.png",
        title="Effect of Positive-Only vs Positive+Negative Prompting (DDIM)",
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = project_root / "outputs" / "task_b"
    summary_path = output_dir / "summary.json"
    analysis_dir = output_dir / "analysis"

    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Loading summary...")
    summary = load_summary(summary_path)
    experiments = summary["experiments"]
    print(f"Loaded {len(experiments)} experiments")

    generate_report_plots(experiments, analysis_dir)

    print(f"\nDone. Plots saved to: {analysis_dir}")


if __name__ == "__main__":
    main()