import torch


class LpLoss(torch.nn.Module):
    def __init__(self, p: int = 2):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - y_pred) ** self.p)
