from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .captioning import evaluate_checkpoint, train #! perque no m'ho troba? mirar julia
except ImportError:
    from captioning import evaluate_checkpoint, train

"""

DESCARREGAR DATASET : 

wget /DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week3/data/train.zip https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip
wget /DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week3/data/annotations.zip https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip

RUN TRAINING: 

python Week3/main.py train --output-dir Week3/runs/baseline --encoder resnet18 --decoder gru --token-level char --epochs 5

- Encoder: resnet18
- Decoder: gru
- Text level: char
- Attention: off

# --------------------------------------------

python Week3/main.py train --output-dir path_on_guardar --encoder (resnet18 / resnet34...) --decoder (gru / lstm) --token-level (char / word) --epochs 5

En teoria funciona tot :)

Prova attention afegir --use-attention  (no sé si va massa bé no ho he provat encara)

SLIDES (borrar):

show one slide with the baseline architecture ans maybe explain dataset
show one slide with the modified architecture and what changed
add a table with metrucs (bleu, rousge....)
add 3 to 5 qualitative examples good prediction, failure case, textheavy image, blurry image and attention result if used ns
discuss one change at a time  encoder depth, decoder type, tokenization level and attention....

"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 3 image captioning experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a captioning model")
    train_parser.add_argument("--data-dir", type=Path, default=Path("Week3/data")) # jo ho tinc així canvieu el path
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--encoder", type=str, default="resnet18")
    train_parser.add_argument("--decoder", type=str, default="gru")
    train_parser.add_argument("--token-level", type=str, default="char")
    train_parser.add_argument("--use-attention", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--max-len", type=int, default=40)
    train_parser.add_argument("--vocab-size", type=int, default=5000)
    train_parser.add_argument("--min-freq", type=int, default=2)
    train_parser.add_argument("--embedding-dim", type=int, default=256)
    train_parser.add_argument("--hidden-dim", type=int, default=512)
    train_parser.add_argument("--pretrained-encoder", action="store_true")
    train_parser.add_argument("--trainable-backbone", action="store_true")
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--limit-train", type=int)
    train_parser.add_argument("--limit-val", type=int)
    train_parser.add_argument("--val-ratio", type=float, default=0.1)
    train_parser.add_argument("--split-seed", type=int, default=42)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved checkpoint")
    eval_parser.add_argument("--data-dir", type=Path, default=Path("Week3/data"))
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--split", type=str, default="val")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--max-examples", type=int)
    eval_parser.add_argument("--output-json", type=Path)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "train":
        summary = train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            encoder_name=args.encoder,
            decoder_type=args.decoder,
            token_level=args.token_level,
            use_attention=args.use_attention,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_len=args.max_len,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            pretrained_encoder=args.pretrained_encoder,
            trainable_backbone=args.trainable_backbone,
            num_workers=args.num_workers,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
        )
        print(json.dumps(summary, indent=2))
        return 0

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_examples=args.max_examples,
    )
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
