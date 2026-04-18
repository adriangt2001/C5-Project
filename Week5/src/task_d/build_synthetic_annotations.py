from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_coco_payload(generations: list[dict], description: str) -> dict:
    images = []
    annotations = []

    for idx, generation in enumerate(generations):
        images.append(
            {
                "file_name": generation["file_name"],
                "vizwiz_url": None,
                "id": idx,
                "text_detected": False,
            }
        )
        annotations.append(
            {
                "caption": generation["caption"],
                "image_id": idx,
                "is_precanned": False,
                "is_rejected": False,
                "id": idx,
                "text_detected": False,
            }
        )

    return {
        "info": {
            "description": description,
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "images": images,
        "annotations": annotations,
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_successful_generations_from_flux_reports(report_paths: list[Path]) -> list[dict]:
    generations = []
    seen_image_paths = set()

    for report_path in report_paths:
        log(f"Reading FLUX report: {report_path}")
        for record in read_jsonl(report_path):
            if record.get("event") != "image_finished":
                continue
            if record.get("status") != "success":
                continue

            image_path_value = record.get("image_path")
            caption = (record.get("caption") or "").strip()
            if not image_path_value or not caption:
                continue

            image_path = resolve_path(image_path_value)
            if image_path is None:
                continue
            if image_path in seen_image_paths:
                continue
            seen_image_paths.add(image_path)

            generations.append(
                {
                    "image_path": image_path,
                    "caption": caption,
                    "theme": record.get("theme"),
                    "source_file_name": record.get("source_file_name"),
                    "seed": record.get("seed"),
                }
            )

    return generations


def load_successful_generations_from_sd_manifest(manifest_path: Path) -> list[dict]:
    payload = read_json(manifest_path)
    generations = payload.get("generations", [])
    successful_generations = []

    for generation in generations:
        if generation.get("status") not in {"success", "skipped_existing"}:
            continue

        image_path_value = generation.get("image_path")
        caption = (generation.get("caption") or "").strip()
        if not image_path_value or not caption:
            continue

        image_path = Path(image_path_value)
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path

        successful_generations.append(
            {
                "image_path": image_path,
                "caption": caption,
                "theme": generation.get("theme"),
                "source_file_name": generation.get("source_file_name"),
                "seed": generation.get("seed"),
            }
        )

    return successful_generations


def create_flux_symlinks(generations: list[dict], output_images_dir: Path) -> list[dict]:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    prepared_generations = []

    for idx, generation in enumerate(generations):
        src_image_path = generation["image_path"]
        if not src_image_path.exists():
            raise FileNotFoundError(f"Generated FLUX image not found: {src_image_path}")

        suffix = src_image_path.suffix.lower() or ".png"
        file_name = f"synthetic_{idx:06d}{suffix}"
        dst_path = output_images_dir / file_name

        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        dst_path.symlink_to(src_image_path.resolve())

        prepared_generations.append(
            {
                **generation,
                "file_name": file_name,
            }
        )

    return prepared_generations


def copy_sd_images(generations: list[dict], output_images_dir: Path) -> list[dict]:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    prepared_generations = []

    for idx, generation in enumerate(generations):
        src_image_path = generation["image_path"]
        if not src_image_path.exists():
            raise FileNotFoundError(f"Generated SD image not found: {src_image_path}")

        suffix = src_image_path.suffix.lower() or ".png"
        file_name = f"synthetic_{idx:06d}{suffix}"
        dst_path = output_images_dir / file_name
        shutil.copy2(src_image_path, dst_path)

        prepared_generations.append(
            {
                **generation,
                "file_name": file_name,
            }
        )

    return prepared_generations


def build_flux_synthetic_annotations(args) -> None:
    report_paths = [resolve_path(path) for path in args.report_paths]
    output_root = resolve_path(args.output_dir)
    if output_root is None:
        raise ValueError("output_dir is required")

    annotations_dir = output_root / "annotations"
    synthetic_images_dir = output_root / "synthetic_flux"
    output_json_path = annotations_dir / "train_synthetic_flux.json"

    log(f"FLUX output root: {output_root}")
    generations = load_successful_generations_from_flux_reports(report_paths)
    if args.limit is not None:
        generations = generations[: int(args.limit)]
        log(f"Applied generation limit: {args.limit}")

    log(f"Found {len(generations)} successful FLUX generated images")
    prepared_generations = create_flux_symlinks(generations, synthetic_images_dir)
    payload = build_coco_payload(
        prepared_generations,
        description="Synthetic training split generated from FLUX reports",
    )
    save_json(output_json_path, payload)

    log(f"Created {len(prepared_generations)} symbolic links in {synthetic_images_dir}")
    log(f"Saved synthetic annotations to {output_json_path}")


def build_sd_synthetic_annotations(args) -> None:
    manifest_path = resolve_path(args.manifest_path)
    output_root = resolve_path(args.output_dir)
    if manifest_path is None:
        raise ValueError("manifest_path is required for sd mode")
    if output_root is None:
        raise ValueError("output_dir is required")

    annotations_dir = output_root / "annotations"
    synthetic_images_dir = output_root / "synthetic_sd"
    output_json_path = annotations_dir / "train_synthetic_sd.json"

    log(f"SD manifest path: {manifest_path}")
    log(f"SD output root: {output_root}")

    generations = load_successful_generations_from_sd_manifest(manifest_path)
    if args.limit is not None:
        generations = generations[: int(args.limit)]
        log(f"Applied generation limit: {args.limit}")

    log(f"Found {len(generations)} successful SD generated images")
    prepared_generations = copy_sd_images(generations, synthetic_images_dir)
    payload = build_coco_payload(
        prepared_generations,
        description="Synthetic training split generated from Stable Diffusion manifest",
    )
    save_json(output_json_path, payload)

    log(f"Copied {len(prepared_generations)} images to {synthetic_images_dir}")
    log(f"Saved synthetic annotations to {output_json_path}")


def run_build_synthetic_annotations(args) -> None:
    input_type = getattr(args, "input_type", None)
    if input_type is None:
        input_type = "sd" if getattr(args, "manifest_path", None) else "flux"

    if input_type == "flux":
        if not getattr(args, "report_paths", None):
            raise ValueError("For input_type='flux', provide report_paths in the config.")
        build_flux_synthetic_annotations(args)
    elif input_type == "sd":
        if not getattr(args, "manifest_path", None):
            raise ValueError("For input_type='sd', provide manifest_path in the config.")
        build_sd_synthetic_annotations(args)
    else:
        raise ValueError("input_type must be either 'flux' or 'sd'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train_synthetic.json from FLUX reports or a Stable Diffusion manifest."
    )
    parser.add_argument(
        "--input_type",
        choices=["flux", "sd"],
        required=True,
        help="Choose 'flux' to read generation_report.jsonl files or 'sd' to read captions_with_images.json.",
    )
    parser.add_argument(
        "--report_paths",
        nargs="*",
        default=None,
        help="One or more FLUX generation_report.jsonl files.",
    )
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Path to captions_with_images.json for Stable Diffusion outputs.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for the synthetic dataset. The script will create synthetic/ and annotations/train_synthetic.json inside it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of successful generations to include.",
    )
    run_build_synthetic_annotations(parser.parse_args())


if __name__ == "__main__":
    main()
