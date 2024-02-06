import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.constants import (
    ABALONE_COLMUNS,
    ABALONE_LABEL_MEAN,
    ABALONE_LABEL_STD,
    ABALONE_MEAN,
    ABALONE_PATH,
    ABALONE_STD,
    DATA_PATH,
    MNIST_MEAN,
    MNIST_STD,
)


def transform_category_to_onehot(data: pd.DataFrame, category: str) -> pd.DataFrame:
    onehot = pd.get_dummies(data[category], prefix=category)
    data = pd.concat([data, onehot], axis=1)
    data = data.drop(category, axis=1)

    return data


class AbaloneDataset(Dataset):
    def __init__(self, mode: str = "train", seed: int = 42, valid_ratio: float = 0.2):
        self.mode = mode

        data = pd.read_csv(ABALONE_PATH, header=None)
        data.columns = ABALONE_COLMUNS
        data = transform_category_to_onehot(data, "Sex")

        self.x = data.drop("Rings", axis=1)
        self.y = data["Rings"]

        if mode == "train":
            self.x, _, self.y, _ = train_test_split(
                self.x, self.y, test_size=valid_ratio, random_state=seed
            )
        else:
            _, self.x, _, self.y = train_test_split(
                self.x, self.y, test_size=valid_ratio, random_state=seed
            )

        self.x = (self.x - ABALONE_MEAN) / ABALONE_STD
        self.x = torch.tensor(self.x.values.astype(float)).float()
        self.y = (self.y - ABALONE_LABEL_MEAN) / ABALONE_LABEL_STD
        self.y = torch.tensor(self.y.values.astype(float)).float().reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_abalone_dataloaders(
    batch_size: int = 32,
    seed: int = 42,
    valid_ratio: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader | list, DataLoader | list]:
    train_dataset = AbaloneDataset(mode="train", seed=seed, valid_ratio=valid_ratio)
    test_dataset = AbaloneDataset(mode="test", seed=seed, valid_ratio=valid_ratio)

    print(f"number of training samples: {len(train_dataset)}")
    print(f"number of testing samples: {len(test_dataset)}")

    if batch_size >= len(train_dataset) or batch_size < 0:
        return [(train_dataset.x, train_dataset.y)], [(test_dataset.x, test_dataset.y)]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, test_dataloader


def get_mnist_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    train_dataset = datasets.MNIST(
        DATA_PATH, train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        DATA_PATH, train=False, download=True, transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # dataset = AbaloneDataset(mode="train")
    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
    # print(len(dataset))
    # print(dataset.x)
    # print(dataset.y)

    # test_dataset = AbaloneDataset(mode="test")

    # print(len(test_dataset))
    # print(test_dataset.x.shape)

    train_loader, val_loader = get_mnist_dataloaders()

    for x, y in train_loader:
        print(x.shape, y.shape)
        print(x)
        print(y)
        break
