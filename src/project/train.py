from typing import Tuple

import torch
from torch.utils.data import DataLoader

from project.data import MyDataset
from project.model import Model


def train(epochs: int = 2, batch_size: int = 16, lr: float = 1e-2) -> None:
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
            # Ensure shapes are [N, 1] for the linear layer
            x = x.view(-1, 1)
            y = y.view(-1, 1)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
        print(f"epoch={epoch+1} loss={epoch_loss:.4f}")


if __name__ == "__main__":
    train()
