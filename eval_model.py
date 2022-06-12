import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model import MLP, PlantsDataset, test


def load_model():
    model = MLP()
    model.load_state_dict(torch.load('./model_out'))
    return model


def load_dev_dataset(batch_size=64):
    plant_dev = PlantsDataset('./data/Plant_1_Generation_Data.csv.dev')
    return DataLoader(plant_dev, batch_size=batch_size)


def make_plot(values):
    build_nums = list(range(1, len(values) + 1))
    plt.xlabel('Build number')
    plt.ylabel('MSE Loss')
    plt.plot(build_nums, values, label='Model MSE Loss over builds')
    plt.legend()
    plt.savefig('trend.png')


def main():
    model = load_model()
    dataloader = load_dev_dataset()

    loss_fn = torch.nn.MSELoss()

    loss = test(dataloader, model, loss_fn)
    with open('evaluation_results.txt', 'a+') as f:
        f.write(f'{str(loss)}\n')
    with open('evaluation_results.txt', 'r') as f:
        values = [float(line) for line in f.readlines() if line]
        make_plot(values)


if __name__ == "__main__":
    main()
