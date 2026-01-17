from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple, Union

import torch
import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Minimal dataset that produces scalar samples.

    - Loads numeric samples from text files under `data_path` (one number per line).
    - If no raw files are present, it generates a small synthetic dataset.
    - Returns pairs `(x, y)` where `y == x` for a simple regression toy task.
    """

    def __init__(self, data_path: Union[str, Path]) -> None:
        self.data_path = Path(data_path)
        self._xs: List[float] = []

        if self.data_path.is_dir():
            for txt in sorted(self.data_path.glob("**/*.txt")):
                for line in txt.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._xs.append(float(line))
                    except ValueError:
                        # Skip non-numeric lines to stay robust
                        continue

        # Fallback: create a tiny synthetic dataset
        if not self._xs:
            self._xs = [float(i) for i in range(100)]

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor([self._xs[index]], dtype=torch.float32)  # shape (1,)
        y = x.clone()  # simple identity target
        return x, y

    def preprocess(self, output_folder: Union[str, Path]) -> None:
        """Preprocess raw data and save to `output_folder`.

        For this toy example, we simply serialize the numeric samples
        to a `dataset.pt` file so downstream steps can load quickly.
        """
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._xs, out_dir / "dataset.pt")


SentimentLabel = Literal["negative", "neutral", "positive"]


class FinancialPhraseBankDataset(Dataset):
    """Financial Phrase Bank dataset loader.

    Expects files like `Sentences_AllAgree.txt`, `Sentences_75Agree.txt`,
    `Sentences_66Agree.txt`, `Sentences_50Agree.txt` in `root_path`.

    Each line format: `sentence@sentiment` where sentiment is one of
    `positive|neutral|negative`.
    """

    AGREEMENTS: Dict[str, str] = {
        "AllAgree": "Sentences_AllAgree.txt",
        "75Agree": "Sentences_75Agree.txt",
        "66Agree": "Sentences_66Agree.txt",
        "50Agree": "Sentences_50Agree.txt",
    }

    SENTIMENT_TO_ID: Dict[SentimentLabel, int] = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    def __init__(
        self,
        root_path: Union[str, Path],
        agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
    ) -> None:
        self.root_path = Path(root_path)
        filename = self.AGEMENTS_FILE(agreement)
        file_path = self.root_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected dataset file not found: {file_path}")

        self.sentences: List[str] = []
        self.labels: List[int] = []

        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # Split by last '@' in case sentence has '@' inside
            if "@" not in line:
                continue
            sent, lab = line.rsplit("@", 1)
            lab = lab.strip().lower()
            if lab not in self.SENTIMENT_TO_ID:
                continue
            self.sentences.append(sent.strip())
            self.labels.append(self.SENTIMENT_TO_ID[lab])

        # Vocabulary built lazily; use preprocess/build_vocab for speed
        self.vocab: Dict[str, int] = {}

    @classmethod
    def AGEMENTS_FILE(cls, agreement: str) -> str:
        # helper to map agreement to filename safely
        fname = cls.AGREEMENTS.get(agreement)
        if not fname:
            raise ValueError(
                f"Invalid agreement '{agreement}'. Choose one of {list(cls.AGREEMENTS.keys())}"
            )
        return fname

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.sentences[index], self.labels[index]

    def build_vocab(self, min_freq: int = 1) -> Dict[str, int]:
        """Builds a token-level vocabulary from the dataset.

        Returns a dict mapping token -> index with special tokens:
        PAD=0, UNK=1, and words start from index 2.
        """
        freq: Dict[str, int] = {}
        for s in self.sentences:
            for tok in self.simple_tokenize(s):
                freq[tok] = freq.get(tok, 0) + 1

        vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for tok, count in sorted(freq.items()):
            if count >= min_freq:
                vocab[tok] = idx
                idx += 1
        self.vocab = vocab
        return vocab

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            self.build_vocab()
        return [self.vocab.get(tok, 1) for tok in self.simple_tokenize(text)]

    def collate_fn(self, batch: Sequence[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function to turn a batch of (text, label) into padded tensors.

        Returns:
            inputs: LongTensor of shape [B, T]
            labels: LongTensor of shape [B]
        """
        encoded: List[List[int]] = [self.encode(text) for text, _ in batch]
        labels: List[int] = [lab for _, lab in batch]
        max_len = max((len(seq) for seq in encoded), default=1)
        padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
        inputs = torch.tensor(padded, dtype=torch.long)
        targets = torch.tensor(labels, dtype=torch.long)
        return inputs, targets

    def preprocess(self, output_folder: Union[str, Path], agreement: str = "AllAgree") -> None:
        """Preprocess and cache encoded dataset for faster training.

        Saves `phrasebank_<agreement>.pt` with a dict containing `vocab`, `inputs`, `labels`.
        """
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.vocab:
            self.build_vocab()
        encoded_inputs = [self.encode(s) for s in self.sentences]
        torch.save(
            {"vocab": self.vocab, "inputs": encoded_inputs, "labels": self.labels},
            out_dir / f"phrasebank_{agreement}.pt",
        )


def preprocess(
    data_path: Union[str, Path],
    output_folder: Union[str, Path],
    agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
) -> None:
    """CLI entry to preprocess either numeric or phrasebank data.

    If `data_path` points to FinancialPhraseBank root containing `Sentences_*.txt`,
    we preprocess the phrasebank; otherwise we fallback to numeric dataset.
    """
    print("Preprocessing data...")
    data_root = Path(data_path)
    has_phrasebank = any(
        (data_root / fname).exists() for fname in FinancialPhraseBankDataset.AGREEMENTS.values()
    )
    if has_phrasebank:
        ds = FinancialPhraseBankDataset(data_root, agreement=agreement)
        ds.preprocess(output_folder, agreement=agreement)
        print(
            f"Saved encoded phrasebank ({agreement}) to {Path(output_folder) / f'phrasebank_{agreement}.pt'}"
        )
    else:
        dataset = MyDataset(data_root)
        dataset.preprocess(output_folder)
        print(f"Saved numeric dataset to {Path(output_folder) / 'dataset.pt'}")


if __name__ == "__main__":
    typer.run(preprocess)
