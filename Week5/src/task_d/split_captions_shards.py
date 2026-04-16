import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def flatten_by_image(captions_by_theme: dict) -> dict:
    image_records = {}
    for theme, items in captions_by_theme.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            file_name = item.get("file_name")
            caption = item.get("caption")
            if not file_name or not caption:
                continue
            if file_name not in image_records:
                image_records[file_name] = {"file_name": file_name, "items": []}
            image_records[file_name]["items"].append(
                {
                    "file_name": file_name,
                    "caption": caption,
                    "theme": theme,
                }
            )
    return image_records


def assign_images_to_shards(image_records: dict, num_shards: int) -> list[list[dict]]:
    shards = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards

    sorted_images = sorted(
        image_records.values(),
        key=lambda record: len(record["items"]),
        reverse=True,
    )

    for image_record in sorted_images:
        shard_idx = min(range(num_shards), key=lambda idx: shard_sizes[idx])
        shards[shard_idx].append(image_record)
        shard_sizes[shard_idx] += len(image_record["items"])

    return shards


def build_shard_payload(source_payload: dict, shard_images: list[dict], shard_index: int, num_shards: int) -> dict:
    captions_by_theme = defaultdict(list)
    flat_captions = []

    for image_record in shard_images:
        for item in image_record["items"]:
            theme = item["theme"]
            captions_by_theme[theme].append(
                {
                    "file_name": item["file_name"],
                    "caption": item["caption"],
                }
            )
            flat_captions.append(item["caption"])

    ordered_themes = sorted(source_payload.get("captions_by_theme", {}).keys())
    ordered_captions_by_theme = {theme: captions_by_theme.get(theme, []) for theme in ordered_themes}

    payload = {
        "source_annotations_path": source_payload.get("source_annotations_path"),
        "seed": source_payload.get("seed"),
        "samples_per_weight": source_payload.get("samples_per_weight"),
        "drop_fallback": source_payload.get("drop_fallback"),
        "unique_captions": source_payload.get("unique_captions"),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "num_sampled_images": len(shard_images),
        "num_captions": len(flat_captions),
        "captions_by_theme": ordered_captions_by_theme,
        "captions": flat_captions,
    }
    return payload


def main(args) -> None:
    input_path = Path(args.input_json)
    payload = read_json(input_path)
    captions_by_theme = payload.get("captions_by_theme")

    if not isinstance(captions_by_theme, dict):
        raise ValueError(f"{input_path} does not contain a 'captions_by_theme' dictionary.")

    image_records = flatten_by_image(captions_by_theme)
    shards = assign_images_to_shards(image_records, args.num_shards)

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    suffix = input_path.suffix or ".json"

    for shard_index, shard_images in enumerate(shards):
        shard_payload = build_shard_payload(payload, shard_images, shard_index, args.num_shards)
        shard_path = output_dir / f"{stem}_shard_{shard_index + 1}_of_{args.num_shards}{suffix}"
        write_json(shard_path, shard_payload)
        print(
            f"Saved shard {shard_index + 1}/{args.num_shards}: "
            f"{shard_payload['num_sampled_images']} images, "
            f"{shard_payload['num_captions']} captions -> {shard_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a captions JSON file into balanced shards by source image.")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    main(parser.parse_args())
