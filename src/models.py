import torch


def init_near_zero(module: torch.nn.Module):
    if type(module) == torch.nn.Linear:
        module.weight.data.normal_(0.0, 0.02)
        module.bias.data.fill_(0)


def init_near_identity(module: torch.nn.Module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.orthogonal(module.weight.data)
        module.bias.data.fill_(0)


class LinearRegressionModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: list[int],
        relu_activation: bool = False,
        initializer: str | None = None,  # near_zero, near_identity or None
    ):
        super(LinearRegressionModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer

        self.layers = torch.nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dim:
            self.layers.append(torch.nn.Linear(prev_dim, dim))
            if relu_activation:
                self.layers.append(torch.nn.ReLU())
            prev_dim = dim

        self.layers.append(torch.nn.Linear(prev_dim, output_dim))

        self.layers.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module):
        if self.initializer == "near_zero":
            init_near_zero(module)
        elif self.initializer == "near_identity":
            init_near_identity(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x
