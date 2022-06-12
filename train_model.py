import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import PlantsDataset, MLP, train, test

default_batch_size = 64
default_epochs = 5

device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_args():
    args_parser = argparse.ArgumentParser(prefix_chars='-')
    args_parser.add_argument('-b', '--batchSize', type=int, default=default_batch_size)
    args_parser.add_argument('-e', '--epochs', type=int, default=default_epochs)

    return args_parser.parse_args()


if __name__ == "__main__":
    args = setup_args()
    batch_size = args.batchSize
    epochs = args.epochs

    print(f"Using {device} device")

    plant_test = PlantsDataset('Plant_1_Generation_data.csv.test')
    plant_train = PlantsDataset('Plant_1_Generation_Data.csv.train')

    input_example = np.array([plant_test.x_train.numpy()[0]])

    train_dataloader = DataLoader(plant_train, batch_size=batch_size)
    test_dataloader = DataLoader(plant_test, batch_size=batch_size)

    for i, (data, labels) in enumerate(train_dataloader):
        print(data.shape, labels.shape)
        print(data, labels)
        break

    model = MLP()
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Done!")

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        last_loss = test(test_dataloader, model, loss_fn)
        print(f'rmse: {last_loss}\n')

    torch.save(model.state_dict(), './model_out')
    print("Model saved in ./model_out file.")
