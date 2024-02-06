import torch


class LpLoss(torch.nn.Module):
    def __init__(self, p: int = 2):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - y_pred) ** self.p)

class AccuracyMetric(torch.nn.Module):
    def __init__(self):
        super(AccuracyMetric, self).__init__()

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred.argmax(dim=1) == y).float())
