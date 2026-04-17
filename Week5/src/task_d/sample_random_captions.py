import argparse
import json
import random
from pathlib import Path


FALLBACK_CAPTION = "quality issues are too severe to recognize visual content."

THEMES = {
    "quality_unclear": [
        "quality issues", "blurry", "too blurry", "cannot tell", "hard to see",
        "too dark", "too bright", "unrecognizable", "not clear",
    ],
    "screens_devices": [
        "screen", "computer", "monitor", "laptop", "windows", "dialog",
        "tv", "television", "keyboard", "phone", "captcha",
    ],
    "documents_text": [
        "receipt", "paper", "label", "text", "card", "book", "coupon",
        "instructions", "poster", "sign", "barcode", "mail", "letter", "package", "manual",
    ],
    "food_drink": [
        "food", "bottle", "drink", "soda", "can", "medicine", "cereal",
        "snack", "meat", "plate", "cup", "popcorn",
    ],
    "clothing_body": [
        "shirt", "sweater", "shoe", "sock", "foot", "feet", "pants",
        "bra", "fabric", "clothing", "dress", "hand", "arm",
    ],
    "household_objects": [
        "vacuum", "dryer", "dishwasher", "toilet", "filter", "chair", "table",
        "desk", "counter", "appliance", "machine", "box",
    ],
    "outdoor_transport": [
        "bus", "car", "truck", "building", "tree", "street", "restaurant",
        "mall", "outdoor", "mailbox",
    ],
}


def read_annotations(path):
    with open(path, "r") as f:
        return json.load(f)


def assign_theme(references):
    reference_text = " ".join(references).lower()
    best_theme = "other"
    best_score = 0
    for theme, keywords in THEMES.items():
        score = sum(keyword in reference_text for keyword in keywords)
        if score > best_score:
            best_theme = theme
            best_score = score
    return best_theme


def build_image_records(annotation_data):
    image_by_id = {image["id"]: image for image in annotation_data["images"]}
    references_by_image = {}

    for annotation in annotation_data["annotations"]:
        image_id = annotation["image_id"]
        references_by_image.setdefault(image_id, []).append(
            annotation["caption"].strip())

    image_records = []
    for image_id, references in references_by_image.items():
        image_info = image_by_id.get(image_id, {})
        image_records.append(
            {
                "image_id": image_id,
                "file_name": image_info.get("file_name"),
                "references": references,
            }
        )

    return image_records


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main(args):
    rng = random.Random(args.seed)

    train_data = read_annotations(args.train_annotations_path)
    image_records = build_image_records(train_data)

    categorized_annotations = {}
    for record in image_records:
        theme = assign_theme(record["references"])
        categorized_annotations.setdefault(theme, []).append(record)

    weights = {
        "quality_unclear": 0,
        "screens_devices": 2,
        "documents_text": 2,
        "food_drink": 1,
        "clothing_body": 2,
        "household_objects": 1,
        "outdoor_transport": 2,
        "other": 2,
    }

    sampled_caption_records = []
    theme_summary = []
    for theme, annotations in categorized_annotations.items():
        weight = weights.get(theme, 1)
        target_captions = weight * args.samples_per_weight

        caption_pool = []
        for record in annotations:
            for caption in record["references"]:
                cleaned_caption = caption.strip()
                if not cleaned_caption:
                    continue
                if args.drop_fallback and cleaned_caption.lower() == FALLBACK_CAPTION:
                    continue
                caption_pool.append(
                    {
                        "theme": theme,
                        "file_name": record["file_name"],
                        "caption": cleaned_caption,
                    }
                )

        n_captions = min(target_captions, len(caption_pool))

        if n_captions == 0:
            theme_summary.append(
                {
                    "theme": theme,
                    "available_images": len(annotations),
                    "available_captions": len(caption_pool),
                    "sampled_captions": 0,
                }
            )
            continue

        theme_samples = rng.sample(caption_pool, n_captions)
        sampled_caption_records.extend(theme_samples)
        theme_summary.append(
            {
                "theme": theme,
                "available_images": len(annotations),
                "available_captions": len(caption_pool),
                "sampled_captions": n_captions,
            }
        )

    if args.unique_captions:
        deduped_records = []
        seen_captions = set()
        for record in sampled_caption_records:
            key = record["caption"]
            if key in seen_captions:
                continue
            seen_captions.add(key)
            deduped_records.append(record)
        sampled_caption_records = deduped_records

    captions = []
    sampled_image_ids = set()
    captions_by_theme = {theme: []
                         for theme in sorted(categorized_annotations)}
    for record in sampled_caption_records:
        captions.append(record["caption"])
        sampled_image_ids.add(record["file_name"])
        captions_by_theme.setdefault(record["theme"], []).append(
            {
                "file_name": record["file_name"],
                "caption": record["caption"],
            }
        )

    payload = {
        "source_annotations_path": args.train_annotations_path,
        "seed": args.seed,
        "samples_per_weight": args.samples_per_weight,
        "drop_fallback": args.drop_fallback,
        "unique_captions": args.unique_captions,
        "num_sampled_images": len(sampled_image_ids),
        "num_captions": len(captions),
        "theme_summary": sorted(theme_summary, key=lambda row: row["theme"]),
        "captions_by_theme": captions_by_theme,
    }

    output_path = Path(args.output_captions_path)
    write_json(output_path, payload)
    print(
        f"Saved {len(captions)} captions from {len(sampled_image_ids)} sampled images to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_annotations_path",
        type=str,
        default="data/annotations/train_filtered.json",
    )
    parser.add_argument(
        "--output_captions_path",
        type=str,
        default="data/generated/captions.json",
    )
    parser.add_argument(
        "--samples_per_weight",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--drop_fallback",
        action="store_true",
        help="Exclude the fallback caption from the exported caption list.",
    )
    parser.add_argument(
        "--unique_captions",
        action="store_true",
        help="Remove duplicate captions while preserving order.",
    )
    args = parser.parse_args()
    main(args)
