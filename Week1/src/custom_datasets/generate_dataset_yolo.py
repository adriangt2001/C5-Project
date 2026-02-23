import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image


# KITTI-MOTS convention:
# pixel value = class_id * 1000 + track_id
# class_id: 1=Car, 2=Pedestrian (commonly)
DEFAULT_CLASS_MAP: Dict[int, int] = {
    1: 0,  # Car -> YOLO class 0
    2: 1,  # Pedestrian -> YOLO class 1
}

DEFAULT_CLASS_NAMES = ["Car", "Pedestrian"]


def read_seqmap(seqmap_path: Path) -> List[str]:
    """
    Reads a seqmap file like val.seqmap and returns list of sequence IDs (e.g., '0000', '0001', ...).
    Common formats:
      - header line + one seq per line
      - or just seq per line
    We'll parse the first token per non-empty, non-comment line that looks like a sequence id.
    """
    seqs = []
    with seqmap_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Many seqmap files have a header like: "name" or "sequence" first line
            # We'll accept tokens that are digits.
            tok = line.split()[0]
            if tok.isdigit():
                seqs.append(tok.zfill(4))
    return sorted(set(seqs))


def load_mask_16bit(path: Path) -> np.ndarray:
    """
    Loads a PNG mask, keeping 16-bit values if present.
    Returns HxW np array of integers.
    """
    img = Image.open(path)
    arr = np.array(img)
    # Some PIL modes can yield uint8 even if stored 16-bit; handle typical uint16 properly:
    if arr.dtype == np.uint8 and img.mode in ("I;16", "I;16B", "I;16L"):
        # Sometimes PIL already returns uint16; but just in case:
        arr = arr.astype(np.uint16)
    return arr


def bbox_from_mask(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns (x1,y1,x2,y2) inclusive-ish pixel bbox from boolean mask.
    x2,y2 are max indices (inclusive). If empty mask, return None.
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    """
    Convert pixel bbox to YOLO normalized format (xc, yc, bw, bh) in [0,1].
    Assumes x2,y2 inclusive; convert to width/height accordingly.
    """
    # +1 for inclusive max coordinate
    bw = (x2 - x1 + 1) / w
    bh = (y2 - y1 + 1) / h
    xc = (x1 + x2 + 1) / 2 / w
    yc = (y1 + y2 + 1) / 2 / h
    return xc, yc, bw, bh


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def convert_sequence(
    seq: str,
    split: str,
    kitti_root: Path,
    out_root: Path,
    class_map: Dict[int, int],
    symlink_images: bool,
    skip_empty: bool,
) -> Tuple[int, int]:
    """
    Converts a single sequence folder:
      training/image_02/<seq>/*.png
      instances/<seq>/*.png
    Produces:
      out/images/<split>/<seq>_<frame>.png
      out/labels/<split>/<seq>_<frame>.txt

    Returns (#images_processed, #labels_written)
    """
    img_dir = kitti_root / "training" / "image_02" / seq
    mask_dir = kitti_root / "instances" / seq

    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    img_files = sorted(img_dir.glob("*.png"))
    n_img, n_lbl = 0, 0

    for img_path in img_files:
        frame = img_path.stem  # '00000'
        mask_path = mask_dir / f"{frame}.png"
        if not mask_path.exists():
            # If missing masks, skip or create empty label
            if skip_empty:
                continue

        # Load image size (cheap)
        with Image.open(img_path) as im:
            w, h = im.size

        # Compose YOLO file base name
        base = f"{seq}_{frame}"
        out_img = out_img_dir / f"{base}.png"
        out_lbl = out_lbl_dir / f"{base}.txt"

        # Copy/symlink image
        link_or_copy(img_path, out_img, symlink_images)
        n_img += 1

        # Create labels
        lines: List[str] = []
        if mask_path.exists():
            mask = load_mask_16bit(mask_path)
            if mask.shape[0] != h or mask.shape[1] != w:
                # Usually sizes match; if not, we still compute bboxes in mask coords
                h_m, w_m = mask.shape[:2]
                h_use, w_use = h_m, w_m
            else:
                h_use, w_use = h, w

            ids = np.unique(mask)
            ids = ids[ids != 0]  # remove background

            for inst_id in ids:
                inst_id = int(inst_id)
                class_id = inst_id // 1000
                if class_id not in class_map:
                    continue  # ignore other classes if present
                yolo_cls = class_map[class_id]

                inst_mask = (mask == inst_id)
                bb = bbox_from_mask(inst_mask)
                if bb is None:
                    continue
                x1, y1, x2, y2 = bb
                xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w_use, h_use)

                # Optional: filter degenerate boxes
                if bw <= 0 or bh <= 0:
                    continue

                lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Write label file (YOLO expects an empty file if no objects)
        if lines or not skip_empty:
            out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
            n_lbl += 1

    return n_img, n_lbl


def write_dataset_yaml(out_root: Path, class_names: List[str]) -> None:
    yaml_path = out_root / "dataset.yaml"
    content = (
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    yaml_path.write_text(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", type=str, required=True,
                    help="Path to KITTI-MOTS root (contains training/, instances/, instances_txt/ ...)")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output dataset folder to create (YOLO format)")
    ap.add_argument("--val_seqmap", type=str, default=None,
                    help="Path to val.seqmap to choose validation sequences. If not provided, uses first sequence as val.")
    ap.add_argument("--symlink_images", action="store_true",
                    help="Symlink images instead of copying (faster, saves disk).")
    ap.add_argument("--skip_empty", action="store_true",
                    help="If set, skip images that have no mask/objects; otherwise create empty label files.")
    ap.add_argument("--car_class_id", type=int, default=1, help="KITTI-MOTS class_id for Car (default 1)")
    ap.add_argument("--ped_class_id", type=int, default=2, help="KITTI-MOTS class_id for Pedestrian (default 2)")
    args = ap.parse_args()

    kitti_root = Path(args.kitti_root)
    out_root = Path(args.out_root)

    ensure_dir(out_root / "images" / "train")
    ensure_dir(out_root / "images" / "val")
    ensure_dir(out_root / "labels" / "train")
    ensure_dir(out_root / "labels" / "val")

    class_map = {
        args.car_class_id: 0,
        args.ped_class_id: 1,
    }

    # Determine sequences
    seq_dirs = sorted([p.name for p in (kitti_root / "training" / "image_02").iterdir() if p.is_dir() and p.name.isdigit()])
    if not seq_dirs:
        raise RuntimeError(f"No sequences found under {kitti_root}/training/image_02")

    if args.val_seqmap:
        val_seqs = read_seqmap(Path(args.val_seqmap))
        if not val_seqs:
            raise RuntimeError(f"Could not parse any sequences from {args.val_seqmap}")
    else:
        val_seqs = [seq_dirs[0]]  # fallback: first sequence as val

    val_set = set(val_seqs)
    train_seqs = [s for s in seq_dirs if s not in val_set]
    # If val contains all, fallback
    if not train_seqs:
        train_seqs = seq_dirs[1:]
        if not train_seqs:
            train_seqs = seq_dirs

    print(f"Found {len(seq_dirs)} sequences total.")
    print(f"Using {len(train_seqs)} train sequences, {len(val_seqs)} val sequences.")
    print(f"Val sequences: {sorted(val_set)}")

    total = {"train": (0, 0), "val": (0, 0)}

    for seq in train_seqs:
        n_img, n_lbl = convert_sequence(
            seq=seq,
            split="train",
            kitti_root=kitti_root,
            out_root=out_root,
            class_map=class_map,
            symlink_images=args.symlink_images,
            skip_empty=args.skip_empty,
        )
        total["train"] = (total["train"][0] + n_img, total["train"][1] + n_lbl)

    for seq in sorted(val_set):
        n_img, n_lbl = convert_sequence(
            seq=seq,
            split="val",
            kitti_root=kitti_root,
            out_root=out_root,
            class_map=class_map,
            symlink_images=args.symlink_images,
            skip_empty=args.skip_empty,
        )
        total["val"] = (total["val"][0] + n_img, total["val"][1] + n_lbl)

    write_dataset_yaml(out_root, DEFAULT_CLASS_NAMES)

    print("\nDone.")
    print(f"Train: images={total['train'][0]}, labels_written={total['train'][1]}")
    print(f"Val:   images={total['val'][0]}, labels_written={total['val'][1]}")
    print(f"YOLO dataset.yaml written to: {out_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()


# python generate_dataset_kittimots_to_yolo.py \
#   --kitti_root dataset/KITTI-MOTS \
#   --out_root dataset/yolo_kittimots \
#   --val_seqmap src/custom_datasets/val.seqmap \
#   --symlink_images