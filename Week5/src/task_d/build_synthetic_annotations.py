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


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_successful_generations(report_paths: list[Path]) -> list[dict]:
    generations = []
    seen_image_paths = set()

    for report_path in report_paths:
        log(f"Reading report: {report_path}")
        for record in read_jsonl(report_path):
            if record.get("event") != "image_finished":
                continue
            if record.get("status") != "success":
                continue

            image_path_value = record.get("image_path")
            caption = record.get("caption")
            if not image_path_value or not caption:
                continue

            image_path = resolve_path(image_path_value)
            if image_path in seen_image_paths:
                continue
            seen_image_paths.add(image_path)

            generations.append(
                {
                    "image_path": image_path,
                    "caption": caption.strip(),
                    "theme": record.get("theme"),
                    "source_file_name": record.get("source_file_name"),
                    "seed": record.get("seed"),
                }
            )

    return generations


def build_coco_payload(generations: list[dict], output_images_dir: Path) -> dict:
    images = []
    annotations = []

    for idx, generation in enumerate(generations):
        file_name = str(generation["image_path"]).split("/")[-1]
        print(file_name, flush=True)

        images.append(
            {
                "file_name": file_name,
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

    payload = {
        "info": {
            "description": "Synthetic training split generated from diffusion models",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "images": images,
        "annotations": annotations,
    }
    return payload


# def copy_images(generations: list[dict], output_images_dir: Path) -> None:
#     output_images_dir.mkdir(parents=True, exist_ok=True)
#     for idx, generation in enumerate(generations):
#         src_image_path = generation["image_path"]
#         if not src_image_path.exists():
#             raise FileNotFoundError(f"Generated image not found: {src_image_path}")
#         suffix = src_image_path.suffix.lower() or ".png"
#         dst_path = output_images_dir / f"synthetic_{idx:06d}{suffix}"
#         shutil.copy2(src_image_path, dst_path)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_build_synthetic_annotations_flux(args) -> None:
    report_paths = [resolve_path(path) for path in args.report_paths]
    output_root = resolve_path(args.output_dir)
    annotations_dir = output_root / "annotations"
    synthetic_images_dir = output_root / "synthetic"
    output_json_path = annotations_dir / "train_synthetic.json"

    log(f"Output root: {output_root}")
    generations = load_successful_generations(report_paths)
    if args.limit is not None:
        generations = generations[: int(args.limit)]
        log(f"Applied generation limit: {args.limit}")

    log(f"Found {len(generations)} successful generated images")
    # copy_images(generations, synthetic_images_dir)

    payload = build_coco_payload(generations, synthetic_images_dir)
    save_json(output_json_path, payload)

    # log(f"Copied {len(generations)} images to {synthetic_images_dir}")
    log(f"Saved synthetic annotations to {output_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a COCO-style train_synthetic.json from generation reports."
    )
    parser.add_argument(
        "--report_paths",
        nargs="+",
        required=True,
        help="One or more generation_report.jsonl files from synthetic image generation runs.",
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
    run_build_synthetic_annotations_flux(parser.parse_args())


if __name__ == "__main__":
    main()
