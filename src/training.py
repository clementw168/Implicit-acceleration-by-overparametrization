import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader|list,
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
    val_dataloader: DataLoader|list,
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
