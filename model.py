import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def hour_to_int(text: str):
    return float(text.replace(':', ''))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x.float())


class PlantsDataset(Dataset):
    def __init__(self, file_name):
        df = pd.read_csv(file_name)

        x = np.array([x[0].split(' ')[1] for x in df.iloc[:, 0:1].values])
        y = df.iloc[:, 3].values

        x_processed = np.array([hour_to_int(h) for h in x], dtype='float32')

        self.x_train = torch.from_numpy(x_processed)
        self.y_train = torch.from_numpy(y)
        self.x_train.type(torch.LongTensor)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx].float(), self.y_train[idx].float()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss (using {loss_fn}): {test_loss:>8f} \n")
    return test_loss
