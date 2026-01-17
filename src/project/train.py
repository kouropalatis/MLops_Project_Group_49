from typing import Tuple

import torch
from torch.utils.data import DataLoader

from project.data import FinancialPhraseBankDataset, MyDataset
from project.model import Model, TextSentimentModel


def train_numeric(epochs: int = 2, batch_size: int = 16, lr: float = 1e-2) -> None:
    dataset = MyDataset("data/raw")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x, y = batch  # type: Tuple[torch.Tensor, torch.Tensor]
            x = x.view(-1, 1)
            y = y.view(-1, 1)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
        print(f"numeric | epoch={epoch+1} loss={epoch_loss:.4f}")


def train_phrasebank(
    root_path: str,
    agreement: str = "AllAgree",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    ds = FinancialPhraseBankDataset(root_path, agreement=agreement)  # e.g., F:\Business Analytics Dk\MLOps\FinancialPhraseBank-v1.0
    vocab = ds.build_vocab(min_freq=1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

    model = TextSentimentModel(vocab_size=len(vocab), embedding_dim=64, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))
        acc = correct / max(total, 1)
        print(f"phrasebank({agreement}) | epoch={epoch+1} loss={epoch_loss:.4f} acc={acc:.3f}")


if __name__ == "__main__":
    # Default run numeric toy to preserve simple behavior
    train_numeric()
