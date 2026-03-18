from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

try:
    from .dataset import (
        SimpleTokenizer,
        VizWizCaptionDataset,
        collate_fn,
        load_annotations,
        split_train_val,
    )
except ImportError:
    from dataset import (
        SimpleTokenizer,
        VizWizCaptionDataset,
        collate_fn,
        load_annotations,
        split_train_val,
    )

# nom --> model 
def _get_torchvision_model(name: str, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    weights = None
    if name == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        return models.resnet18(weights=weights)
    if name == "resnet34":
        if pretrained:
            weights = models.ResNet34_Weights.DEFAULT
        return models.resnet34(weights=weights)
    if name == "resnet50":
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        return models.resnet50(weights=weights)
    if name == "vgg16":
        if pretrained:
            weights = models.VGG16_Weights.DEFAULT
        return models.vgg16(weights=weights)
    if name == "vgg19":
        if pretrained:
            weights = models.VGG19_Weights.DEFAULT
        return models.vgg19(weights=weights)
    raise ValueError(f"Unsupported encoder: {name}")

# definir encoder 
class Encoder(nn.Module):
    def __init__(
        self,
        name: str = "resnet18",
        hidden_dim: int = 512,
        pretrained: bool = False,
        trainable_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.name = name.lower()
        backbone = _get_torchvision_model(self.name, pretrained=pretrained) # passar nom del model, dir si pretrained o no

        if self.name.startswith("resnet"):
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
            backbone_dim = 512 if self.name in {"resnet18", "resnet34"} else 2048
        else:
            self.feature_extractor = backbone.features
            backbone_dim = 512

        self.project = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = trainable_backbone

    # forwarad encoder 
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.feature_extractor(images)
        feats = self.project(feats)
        spatial = feats.flatten(2).transpose(1, 2)
        global_feat = self.avg_pool(feats).flatten(1)
        return spatial, global_feat

# prova attention <3
class Attention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder_proj = nn.Linear(hidden_dim, hidden_dim) 
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(
        self, spatial_feats: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expanded_hidden = self.decoder_proj(hidden_state).unsqueeze(1)
        energy = torch.tanh(self.encoder_proj(spatial_feats) + expanded_hidden)
        scores = self.score(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(spatial_feats * weights.unsqueeze(-1), dim=1)
        return context, weights

# definir decoder
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        decoder_type: str = "gru",
        use_attention: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder_type = decoder_type.lower()
        self.use_attention = use_attention

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        rnn_input_dim = embedding_dim + hidden_dim
        rnn_cls = nn.GRUCell if self.decoder_type == "gru" else nn.LSTMCell # gru o lstm :)
        self.rnn = rnn_cls(rnn_input_dim, hidden_dim) # rnn 
        self.attention = Attention(hidden_dim) if use_attention else None # test per la attention 
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.init_h = nn.Linear(hidden_dim, hidden_dim)
        self.init_c = nn.Linear(hidden_dim, hidden_dim) 

    def init_hidden(
        self, global_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = torch.tanh(self.init_h(global_feat))
        if self.decoder_type == "lstm":
            cell = torch.tanh(self.init_c(global_feat)) # perque lstm cell state 
            return hidden, cell
        return hidden, None

    def _step(
        self,
        token_ids: torch.Tensor,
        spatial_feats: torch.Tensor,
        global_feat: torch.Tensor,
        hidden: torch.Tensor,
        cell: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        embedded = self.embedding(token_ids)
        if self.attention is not None:
            context, _ = self.attention(spatial_feats, hidden)
        else:
            context = global_feat

        rnn_input = torch.cat([embedded, context], dim=-1)
        if self.decoder_type == "lstm":
            hidden, cell = self.rnn(rnn_input, (hidden, cell))
        else:
            hidden = self.rnn(rnn_input, hidden)
        logits = self.output(hidden)
        return logits, hidden, cell

    def forward(
        self,
        spatial_feats: torch.Tensor,
        global_feat: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden, cell = self.init_hidden(global_feat)
        logits_per_step: List[torch.Tensor] = []
        for step in range(input_ids.size(1)):
            logits, hidden, cell = self._step(
                input_ids[:, step], spatial_feats, global_feat, hidden, cell
            )
            logits_per_step.append(logits.unsqueeze(1))
        return torch.cat(logits_per_step, dim=1)
    
    # generar captions

    # input --> spatial_feats, global_feat, sos_id, eos_id, max_len
    # output --> generated token ids

    def generate(
        self,
        spatial_feats: torch.Tensor,
        global_feat: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        hidden, cell = self.init_hidden(global_feat)
        token_ids = torch.full(
            (global_feat.size(0),),
            fill_value=sos_id,
            dtype=torch.long,
            device=global_feat.device,
        )
        sequences = []
        finished = torch.zeros(global_feat.size(0), dtype=torch.bool, device=global_feat.device)

        for _ in range(max_len):
            logits, hidden, cell = self._step(
                token_ids, spatial_feats, global_feat, hidden, cell
            )
            token_ids = logits.argmax(dim=-1)
            token_ids = torch.where(
                finished,
                torch.full_like(token_ids, eos_id),
                token_ids,
            )
            sequences.append(token_ids.unsqueeze(1))
            finished |= token_ids.eq(eos_id)
            if finished.all():
                break
        if not sequences:
            return torch.empty(global_feat.size(0), 0, dtype=torch.long, device=global_feat.device)
        return torch.cat(sequences, dim=1)

#! aixo ha de funcionar com el notebook
class CaptioningModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        spatial_feats, global_feat = self.encoder(images)
        return self.decoder(spatial_feats, global_feat, input_ids)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        self.eval()
        spatial_feats, global_feat = self.encoder(images)
        return self.decoder.generate(
            spatial_feats=spatial_feats,
            global_feat=global_feat,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=max_len,
        )


def build_model(
    vocab_size: int,
    encoder_name: str,
    decoder_type: str,
    hidden_dim: int = 512,
    embedding_dim: int = 256,
    use_attention: bool = False,
    pretrained_encoder: bool = False,
    trainable_backbone: bool = False,
) -> CaptioningModel:
    encoder = Encoder(
        name=encoder_name,
        hidden_dim=hidden_dim,
        pretrained=pretrained_encoder,
        trainable_backbone=trainable_backbone,
    )
    decoder = Decoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        decoder_type=decoder_type,
        use_attention=use_attention,
    )
    return CaptioningModel(encoder=encoder, decoder=decoder)

# contar ngrams per a les metricas
def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i : i + n])
        counts[key] = counts.get(key, 0) + 1
    return counts

# calcular BLEU score, codi del github
def _bleu_score(
    predictions: Sequence[str], references: Sequence[Sequence[str]], max_order: int
) -> float:
    if not predictions:
        return 0.0

    precisions = []
    pred_len = 0
    ref_len = 0

    for order in range(1, max_order + 1):
        overlap_total = 0
        pred_total = 0
        for pred, refs in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens_list = [ref.split() for ref in refs if ref.strip()]
            pred_len += len(pred_tokens) if order == 1 else 0
            if order == 1 and ref_tokens_list:
                ref_len += min(
                    ref_tokens_list, key=lambda ref: abs(len(ref) - len(pred_tokens))
                ).__len__()

            pred_counts = _ngram_counts(pred_tokens, order)
            max_ref_counts: Dict[Tuple[str, ...], int] = {}
            for ref_tokens in ref_tokens_list:
                ref_counts = _ngram_counts(ref_tokens, order)
                for key, value in ref_counts.items():
                    max_ref_counts[key] = max(max_ref_counts.get(key, 0), value)

            overlap_total += sum(
                min(count, max_ref_counts.get(key, 0))
                for key, count in pred_counts.items()
            )
            pred_total += max(len(pred_tokens) - order + 1, 0)

        precisions.append((overlap_total + 1e-9) / (pred_total + 1e-9))

    if pred_len == 0:
        return 0.0
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_order)

# calcular ROUGE-L, codi del github
def _lcs_length(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    rows = len(seq_a) + 1
    cols = len(seq_b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]

# calcular ROUGE-L, codi del github
def _rouge_l(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    if not predictions:
        return 0.0
    scores = []
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.split()
        best = 0.0
        for ref in refs:
            ref_tokens = ref.split()
            if not pred_tokens or not ref_tokens:
                continue
            lcs = _lcs_length(pred_tokens, ref_tokens)
            precision = lcs / len(pred_tokens)
            recall = lcs / len(ref_tokens)
            if precision + recall == 0:
                score = 0.0
            else:
                score = 2 * precision * recall / (precision + recall)
            best = max(best, score)
        scores.append(best)
    return sum(scores) / len(scores)

# calcular METEOR-lite, codi del github
def _meteor_lite(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    if not predictions:
        return 0.0
    scores = []
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.split()
        best = 0.0
        for ref in refs:
            ref_tokens = ref.split()
            if not pred_tokens or not ref_tokens:
                continue

            matches = []
            used = set()
            for idx, token in enumerate(pred_tokens):
                for ref_idx, ref_token in enumerate(ref_tokens):
                    if ref_idx in used:
                        continue
                    if token == ref_token:
                        used.add(ref_idx)
                        matches.append((idx, ref_idx))
                        break

            m = len(matches)
            if m == 0:
                continue

            precision = m / len(pred_tokens)
            recall = m / len(ref_tokens)
            f_mean = (10 * precision * recall) / (recall + 9 * precision + 1e-9)

            chunks = 1
            for i in range(1, len(matches)):
                prev = matches[i - 1]
                cur = matches[i]
                if cur[0] != prev[0] + 1 or cur[1] != prev[1] + 1:
                    chunks += 1

            penalty = 0.5 * (chunks / m) ** 3
            best = max(best, f_mean * (1 - penalty))
        scores.append(best)
    return sum(scores) / len(scores)

# calcular metriques per captions
def evaluate_captions(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> Dict[str, float]:
    return {
        "bleu1": _bleu_score(predictions, references, max_order=1),
        "bleu2": _bleu_score(predictions, references, max_order=2),
        "rougeL": _rouge_l(predictions, references),
        "meteor": _meteor_lite(predictions, references),
    }

# train epoch com el notebook pero canviant petites coses 
def _run_epoch(
    model: CaptioningModel,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0

    for batch in tqdm(dataloader, leave=False):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        logits = model(images, input_ids)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)

# predir captions + calcular metriques per cada batch
@torch.no_grad()
def _predict(
    model: CaptioningModel,
    dataloader: DataLoader,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    max_len: int,
    max_batches: Optional[int] = None,
) -> Tuple[List[str], List[List[str]], List[Dict[str, object]]]:
    model.eval()
    predictions: List[str] = []
    references: List[List[str]] = []
    qualitative: List[Dict[str, object]] = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device)
        generated = model.generate(
            images=images,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            max_len=max_len,
        )

        for idx in range(generated.size(0)):
            pred = tokenizer.decode(generated[idx].tolist())
            refs = [ref for ref in batch["references"][idx] if ref.strip()]
            predictions.append(pred)
            references.append(refs if refs else [""])
            qualitative.append(
                {
                    "file_name": batch["file_names"][idx],
                    "prediction": pred,
                    "references": refs,
                }
            )
    return predictions, references, qualitative

#! train loop maco, mirar si peta al validation
def train(
    data_dir: Path,
    output_dir: Path,
    encoder_name: str = "resnet18",
    decoder_type: str = "gru",
    token_level: str = "char",
    use_attention: bool = False,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_len: int = 40,
    vocab_size: int = 5000,
    min_freq: int = 2,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    pretrained_encoder: bool = False,
    trainable_backbone: bool = False,
    num_workers: int = 0,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> Dict[str, object]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_annotations(data_dir / "annotations" / "train.json")
    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=val_ratio,
        seed=split_seed,
    )
    if limit_train is not None:
        train_samples = train_samples[:limit_train]
    if limit_val is not None:
        val_samples = val_samples[:limit_val]

    train_captions = [
        caption for sample in train_samples for caption in sample.captions if caption.strip()
    ]
    tokenizer = SimpleTokenizer(
        token_level=token_level,
        min_freq=min_freq,
        vocab_size=vocab_size,
    )
    tokenizer.build_vocab(train_captions)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    train_dataset = VizWizCaptionDataset(
        data_dir=data_dir,
        samples=train_samples,
        tokenizer=tokenizer,
        max_len=max_len,
        training=True,
    )
    val_dataset = VizWizCaptionDataset(
        data_dir=data_dir,
        samples=val_samples,
        tokenizer=tokenizer,
        max_len=max_len,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        vocab_size=tokenizer.vocab_size_actual,
        encoder_name=encoder_name,
        decoder_type=decoder_type,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        use_attention=use_attention,
        pretrained_encoder=pretrained_encoder,
        trainable_backbone=trainable_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[Dict[str, object]] = []
    best_score = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = _run_epoch(model, val_loader, None, criterion, device)
        predictions, references, qualitative = _predict(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_len=max_len,
        )
        metrics = evaluate_captions(predictions, references)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }
        history.append(epoch_record)

        score = metrics["meteor"] + metrics["rougeL"] + metrics["bleu2"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "encoder_name": encoder_name,
                        "decoder_type": decoder_type,
                        "token_level": token_level,
                        "use_attention": use_attention,
                        "max_len": max_len,
                        "vocab_size": tokenizer.vocab_size_actual,
                        "hidden_dim": hidden_dim,
                        "embedding_dim": embedding_dim,
                    },
                },
                best_path,
            )
            (output_dir / "val_predictions.json").write_text(
                json.dumps(qualitative[:50], indent=2)
            )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"BLEU1={metrics['bleu1']:.4f} "
            f"BLEU2={metrics['bleu2']:.4f} "
            f"ROUGE-L={metrics['rougeL']:.4f} "
            f"METEOR={metrics['meteor']:.4f}"
        )

    summary = {
        "best_checkpoint": str(best_path),
        "history": history,
        "config": {
            "encoder_name": encoder_name,
            "decoder_type": decoder_type,
            "token_level": token_level,
            "use_attention": use_attention,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_len": max_len,
            "val_ratio": val_ratio,
            "split_seed": split_seed,
        },
    }
    (output_dir / "history.json").write_text(json.dumps(summary, indent=2))
    return summary

# load model 
def load_checkpoint(
    checkpoint_path: Path,
    tokenizer_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[CaptioningModel, SimpleTokenizer, Dict[str, object]]:
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    if tokenizer_path is None:
        tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    tokenizer = SimpleTokenizer.load(Path(tokenizer_path))

    model = build_model(
        vocab_size=tokenizer.vocab_size_actual,
        encoder_name=config["encoder_name"],
        decoder_type=config["decoder_type"],
        hidden_dim=config["hidden_dim"],
        embedding_dim=config["embedding_dim"],
        use_attention=config["use_attention"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, config

# evaluar 10% del training --> val
@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    split: str = "val",
    batch_size: int = 32,
    num_workers: int = 0,
    max_examples: Optional[int] = None,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_checkpoint(checkpoint_path, device=device)
    samples = load_annotations(Path(data_dir) / "annotations" / f"{split}.json")
    if max_examples is not None:
        samples = samples[:max_examples]

    dataset = VizWizCaptionDataset(
        data_dir=Path(data_dir),
        samples=samples,
        tokenizer=tokenizer,
        max_len=config["max_len"],
        training=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    predictions, references, qualitative = _predict(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=device,
        max_len=config["max_len"],
    )
    metrics = evaluate_captions(predictions, references)
    return {
        "metrics": metrics,
        "examples": qualitative[:50],
        "config": config,
    }
