import json
import argparse
import random


def read_annotations(path):
    with open(path, "r") as f:
        annotations = json.load(f)
    return annotations


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


def main(args):
    # read train annotations
    train_annotations = read_annotations(args.train_annotations_path)

    # categorize annotations
    categorized_annotations = {}
    for annotation in train_annotations:
        theme = assign_theme(annotation["references"])
        if theme not in categorized_annotations:
            categorized_annotations[theme] = []
        categorized_annotations[theme].append(annotation)


    # select random samples weighted by category
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

    # sample random annotations weighted by category
    sampled_annotations = []
    for theme, annotations in categorized_annotations.items():
        weight = weights.get(theme, 1)
        n_annotations = weight * 100  
        # sample 100 random annotations per category, weighted by importance
        sampled_annotations.extend(random.sample(annotations, min(n_annotations, len(annotations))))


    # generate captions and prompts
    captions = {}


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations_path", type=str,
                        default="data/annotations/train_filtered.json")
    parser.add_argument("--output_captions_path", type=str,
                        default="data/generated/captions.json")
    parser.add_argument("--output_prompts_path", type=str,
                        default="data/generated/prompts.json")
    args = parser.parse_args()
    main(args)
