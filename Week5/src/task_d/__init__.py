from .generate_images_flux import run_task_d_flux
from .build_synthetic_annotations import run_build_synthetic_annotations

run_task_d = run_task_d_flux

__all__ = ["run_task_d_flux", "run_build_synthetic_annotations"]
