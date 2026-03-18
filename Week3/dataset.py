from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] # PAD: padding, SOS: start of sequence, EOS: end of sequence, UNK: unknown token
_WORD_RE = re.compile(r"\w+|[^\w\s]") # regex per tokenize words and punctuation

# tret del github de les diapos
@dataclass
class VizWizSample:
    image_id: int
    file_name: str
    split: str
    captions: List[str]
    text_detected: bool

# simple tokenizer -->  tokenizes text at char, word or subword level
class SimpleTokenizer:
    def __init__(
        self,
        token_level: str = "char",
        min_freq: int = 2,
        vocab_size: int = 5000,
    ) -> None:
        self.token_level = token_level
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.tokens: List[str] = list(_SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = {
            token: idx for idx, token in enumerate(self.tokens)
        }
        self.bpe_merges: List[Tuple[str, str]] = []

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def sos_id(self) -> int:
        return self.token_to_id["<SOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def vocab_size_actual(self) -> int:
        return len(self.tokens)

    def _word_tokenize(self, text: str) -> List[str]:
        return _WORD_RE.findall(text.strip()) # extreu paraules

    def _char_tokenize(self, text: str) -> List[str]:
        return list(text.rstrip("\n")) # extreu lletres

    def _get_stats(
        self, vocab: Dict[Tuple[str, ...], int]
    ) -> Counter[Tuple[str, str]]:
        pairs: Counter[Tuple[str, str]] = Counter()
        for symbols, freq in vocab.items():
            if len(symbols) < 2:
                continue
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        merged_token = pair[0] + pair[1]
        new_vocab: Dict[Tuple[str, ...], int] = {}
        for symbols, freq in vocab.items():
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(merged_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_vocab[tuple(new_symbols)] = freq
        return new_vocab

    def _train_bpe(self, captions: Sequence[str]) -> None:
        word_freqs = Counter()
        for caption in captions:
            word_freqs.update(word.lower() for word in self._word_tokenize(caption))

        vocab = {
            tuple(list(word) + ["</w>"]): freq
            for word, freq in word_freqs.items()
            if word
        }

        base_tokens = set()
        for word in word_freqs:
            for char in word:
                base_tokens.add(char)

        learned_tokens = set(base_tokens)
        max_merges = max(self.vocab_size - len(_SPECIAL_TOKENS) - len(base_tokens), 0)
        self.bpe_merges = []

        for _ in range(max_merges):
            stats = self._get_stats(vocab)
            if not stats:
                break
            pair, freq = stats.most_common(1)[0]
            if freq < self.min_freq:
                break
            vocab = self._merge_vocab(pair, vocab)
            merged = pair[0] + pair[1]
            self.bpe_merges.append(pair)
            learned_tokens.add(merged)

        if not learned_tokens:
            self.bpe_merges = []

    def _apply_bpe(self, word: str) -> List[str]:
        if not word:
            return []
        symbols = list(word.lower()) + ["</w>"]
        for left, right in self.bpe_merges:
            merged = left + right
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == left and symbols[i + 1] == right:
                    symbols = symbols[:i] + [merged] + symbols[i + 2 :]
                else:
                    i += 1

        pieces: List[str] = []
        for idx, symbol in enumerate(symbols):
            if symbol == "</w>":
                continue
            if symbol.endswith("</w>"):
                symbol = symbol[:-4]
            if idx < len(symbols) - 2:
                pieces.append(f"{symbol}@@")
            else:
                pieces.append(symbol)
        return pieces

    def tokenize(self, text: str) -> List[str]:
        if self.token_level == "char":
            return self._char_tokenize(text)
        if self.token_level == "word":
            return self._word_tokenize(text.lower())
        if self.token_level == "subword":
            pieces: List[str] = []
            for word in self._word_tokenize(text):
                if re.fullmatch(r"\w+", word):
                    pieces.extend(self._apply_bpe(word))
                else:
                    pieces.append(word)
            return pieces
        raise ValueError(f"Unsupported token level: {self.token_level}")

    def build_vocab(self, captions: Sequence[str]) -> None:
        if self.token_level == "subword":
            self._train_bpe(captions)
            counter = Counter()
            for caption in captions:
                counter.update(self.tokenize(caption))
            units = [
                token
                for token, freq in counter.most_common(self.vocab_size)
                if freq >= self.min_freq
            ]
        else:
            counter = Counter()
            for caption in captions:
                counter.update(self.tokenize(caption))
            units = [
                token
                for token, freq in counter.most_common(self.vocab_size)
                if freq >= self.min_freq
            ]

        self.tokens = list(_SPECIAL_TOKENS)
        for token in units:
            if token not in self.token_to_id and token not in self.tokens:
                self.tokens.append(token)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}

    def encode(self, text: str, max_len: int) -> List[int]:
        token_ids = [self.sos_id]
        for token in self.tokenize(text):
            token_ids.append(self.token_to_id.get(token, self.unk_id))
        token_ids.append(self.eos_id)
        token_ids = token_ids[:max_len]
        if len(token_ids) < max_len:
            token_ids.extend([self.pad_id] * (max_len - len(token_ids)))
        else:
            token_ids[-1] = self.eos_id
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens: List[str] = []
        for token_id in token_ids:
            token = self.tokens[token_id]
            if token in {"<PAD>", "<SOS>"}:
                continue
            if token == "<EOS>":
                break
            tokens.append(token)

        if self.token_level == "char":
            return "".join(tokens).strip()
        if self.token_level == "word":
            text = " ".join(tokens)
            text = re.sub(r"\s+([,.!?;:])", r"\1", text)
            return text.strip()

        words: List[str] = []
        current = ""
        for token in tokens:
            if token.endswith("@@"):
                current += token[:-2]
            else:
                current += token
                words.append(current)
                current = ""
        if current:
            words.append(current)
        text = " ".join(words)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text.strip()

    def save(self, path: Path) -> None:
        payload = {
            "token_level": self.token_level,
            "min_freq": self.min_freq,
            "vocab_size": self.vocab_size,
            "tokens": self.tokens,
            "bpe_merges": self.bpe_merges,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":
        payload = json.loads(path.read_text())
        tokenizer = cls(
            token_level=payload["token_level"],
            min_freq=payload["min_freq"],
            vocab_size=payload["vocab_size"],
        )
        tokenizer.tokens = payload["tokens"]
        tokenizer.token_to_id = {
            token: idx for idx, token in enumerate(tokenizer.tokens)
        }
        tokenizer.bpe_merges = [tuple(pair) for pair in payload.get("bpe_merges", [])]
        return tokenizer

# load el dataset i processar les anotacions
def load_annotations(annotation_path: Path) -> List[VizWizSample]:
    payload = json.loads(annotation_path.read_text())
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    split = annotation_path.stem

    captions_by_image: Dict[int, List[str]] = defaultdict(list)
    for annotation in annotations:
        if annotation.get("is_rejected"):
            continue
        caption = annotation.get("caption", "").strip()
        if caption:
            captions_by_image[annotation["image_id"]].append(caption)

    samples: List[VizWizSample] = []
    for image in images:
        samples.append(
            VizWizSample(
                image_id=image["id"],
                file_name=image["file_name"],
                split=split,
                captions=captions_by_image.get(image["id"], []),
                text_detected=bool(image.get("text_detected", False)),
            )
        )
    return samples

# train/val split --> 10% del training per val
def split_train_val(
    samples: Sequence[VizWizSample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[VizWizSample], List[VizWizSample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_at = int(len(indices) * (1.0 - val_ratio))

    train_samples = [samples[idx] for idx in indices[:split_at]]
    val_samples = [samples[idx] for idx in indices[split_at:]]
    return train_samples, val_samples

# load les imatges i les captions
class VizWizCaptionDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        samples: Sequence[VizWizSample],
        tokenizer: SimpleTokenizer,
        max_len: int = 40,
        training: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.training = training
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _resolve_image_path(self, sample: VizWizSample) -> Path:
        candidates = [
            self.data_dir / sample.split / sample.file_name,
            self.data_dir / "train" / sample.file_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image_path = self._resolve_image_path(sample)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        caption = ""
        if sample.captions:
            caption = random.choice(sample.captions) if self.training else sample.captions[0]

        encoded = self.tokenizer.encode(caption, max_len=self.max_len)
        return {
            "image": image_tensor,
            "input_ids": torch.tensor(encoded[:-1], dtype=torch.long),
            "target_ids": torch.tensor(encoded[1:], dtype=torch.long),
            "caption": caption,
            "references": list(sample.captions),
            "file_name": sample.file_name,
            "image_id": sample.image_id,
        }


def collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    return {
        "images": images,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "captions": [item["caption"] for item in batch],
        "references": [item["references"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
    }
