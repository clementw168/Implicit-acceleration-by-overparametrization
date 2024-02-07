import glob
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.constants import (ABALONE_COLMUNS, ABALONE_LABEL_MEAN,
                           ABALONE_LABEL_STD, ABALONE_MEAN, ABALONE_PATH,
                           ABALONE_STD, DATA_PATH, ETHANOL_LABEL_MEAN,
                           ETHANOL_LABEL_STD, ETHANOL_MEAN, ETHANOL_PATH,
                           ETHANOL_STD, MNIST_MEAN, MNIST_STD)


def transform_category_to_onehot(data: pd.DataFrame, category: str) -> pd.DataFrame:
    onehot = pd.get_dummies(data[category], prefix=category)
    data = pd.concat([data, onehot], axis=1)
    data = data.drop(category, axis=1)

    return data


def get_abalone_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(ABALONE_PATH, header=None)
    data.columns = ABALONE_COLMUNS
    data = transform_category_to_onehot(data, "Sex")

    return data.drop("Rings", axis=1), data["Rings"]


def get_ethanol_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    all_files = glob.glob(os.path.join(ETHANOL_PATH, "*.dat"))

    df_from_each_file = (
        pd.read_csv(f, sep=r"\s+", index_col=0, header=None) for f in all_files
    )
    df = pd.concat(df_from_each_file, sort=True)

    # Seperate feature and value in each cell eg. 1:15596.16 --> 15596.16
    for col in df.columns.values:
        df[col] = df[col].apply(lambda x: float(str(x).split(":")[1]))

    # Make Index(Gas type) a column and reset index to original
    df = df.rename_axis("Gas").reset_index()

    df["concentration"] = df["Gas"].apply(lambda x: float(x.split(";")[1]))

    df["Gas"] = df["Gas"].apply(lambda x: int(x.split(";")[0]))

    # Sort by Gas and reindex
    df.sort_values(by=["Gas"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    ethanol = df[df["Gas"] == 1]
    ethanol = ethanol.drop(columns=["Gas"])

    return ethanol.drop("concentration", axis=1), ethanol["concentration"]


def preprocess_tabular_data(
    x: pd.DataFrame, y: pd.Series, is_abalone: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    if is_abalone:
        return (x - ABALONE_MEAN) / ABALONE_STD, (
            y - ABALONE_LABEL_MEAN
        ) / ABALONE_LABEL_STD
    else:
        return (x - ETHANOL_MEAN) / ETHANOL_STD, (y - ETHANOL_LABEL_MEAN) / ETHANOL_LABEL_STD



class TabularDataset(Dataset):
    def __init__(self, is_abalone:bool=True, mode: str = "train", seed: int = 42, valid_ratio: float = 0.2):
        self.mode = mode
        self.is_abalone = is_abalone

        self.x, self.y = get_abalone_dataframe() if is_abalone else get_ethanol_dataframe()

        if mode == "train":
            self.x, _, self.y, _ = train_test_split(
                self.x, self.y, test_size=valid_ratio, random_state=seed
            )
        else:
            _, self.x, _, self.y = train_test_split(
                self.x, self.y, test_size=valid_ratio, random_state=seed
            )

        self.x, self.y = preprocess_tabular_data(self.x, self.y, is_abalone)

        self.x = torch.tensor(self.x.values.astype(float)).float()
        self.y = torch.tensor(self.y.values.astype(float)).float().reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_tabular_dataloaders(
    batch_size: int = 32,
    is_abalone: bool = True,
    seed: int = 42,
    valid_ratio: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader | list, DataLoader | list]:
    train_dataset = TabularDataset(is_abalone=is_abalone, mode="train", seed=seed, valid_ratio=valid_ratio)
    test_dataset = TabularDataset(is_abalone=is_abalone, mode="test", seed=seed, valid_ratio=valid_ratio)

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
    train_loader, val_loader = get_mnist_dataloaders()

    for x, y in train_loader:
        print(x.shape, y.shape)
        print(x)
        print(y)
        break
