from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader | list,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_tqdm: bool = False,
) -> list[float]:
    model.train()
    loss_accumulator = []

    loader = tqdm(train_dataloader) if use_tqdm else train_dataloader

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        loss_accumulator.append(loss.item())

    return loss_accumulator


def val_loop(
    model: torch.nn.Module,
    val_dataloader: DataLoader | list,
    loss_fn: torch.nn.Module,
    metric_fn: torch.nn.Module,
    device: torch.device,
    use_tqdm: bool = False,
) -> tuple[float, float]:
    model.eval()

    loss_accumulator = 0
    metric_accumulator = 0

    loader = tqdm(val_dataloader) if use_tqdm else val_dataloader

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss_accumulator += loss_fn(y_pred, y).item()
            metric_accumulator += metric_fn(y_pred, y).item()

    return loss_accumulator / len(val_dataloader), metric_accumulator / len(
        val_dataloader
    )


def lr_grid(
    model: torch.nn.Module,
    train_dataloader: DataLoader|list,
    val_dataloader: DataLoader|list,
    loss_fn: torch.nn.Module,
    metric_fn: torch.nn.Module,
    lr_list: list[float],
    epochs: int,
    device: torch.device,
    optimizer_type: str = "SGD"
) -> tuple[list[float], list[float], list[float]]:
    best_train_loss_list = []
    best_val_loss_list = []
    best_val_metric_list = []

    for lr in tqdm(lr_list):
        train_loss_list: list[float] = []
        val_loss_list: list[float] = []
        val_metric_list: list[float] = []

        model_ = deepcopy(model)
        optimizer = torch.optim.SGD(model_.parameters(), lr=lr) if optimizer_type == "SGD" else torch.optim.Adam(model_.parameters(), lr=lr)

        for _ in range(epochs):
            train_loss = train_loop(
                model_, train_dataloader, loss_fn, optimizer, device, use_tqdm=False
            )
            val_loss, val_metric = val_loop(
                model_, val_dataloader, loss_fn, metric_fn, device, use_tqdm=False
            )

            train_loss_list.extend(train_loss)
            val_loss_list.append(val_loss)
            val_metric_list.append(val_metric)

        best_train_loss_list.append(train_loss_list)
        best_val_loss_list.append(val_loss_list)
        best_val_metric_list.append(val_metric_list)

    best_metric = [min(l) for l in best_val_metric_list]
    best_lr_index = best_metric.index(min(best_metric))


    return best_train_loss_list[best_lr_index], best_val_loss_list[best_lr_index], best_val_metric_list[best_lr_index]
