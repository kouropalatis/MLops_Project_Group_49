from pathlib import Path
from typing import List, Tuple, Union

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


def preprocess(data_path: Union[str, Path], output_folder: Union[str, Path]) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
