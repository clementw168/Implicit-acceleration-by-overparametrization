import torch


def init_near_zero(module: torch.nn.Module):
    if type(module) == torch.nn.Linear:
        module.weight.data.normal_(0.0, 0.02)
        module.bias.data.fill_(0)


def init_near_identity(module: torch.nn.Module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.orthogonal_(module.weight.data)
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


class BasicCNN(torch.nn.Module):
    def __init__(self, input_size: tuple[int, int, int], num_classes: int):
        super(BasicCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            input_size[0], 4, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        self.intermediate_shape = (input_size[1] // 8) * (input_size[2] // 8)

        self.fc1 = torch.nn.Linear(16*self.intermediate_shape, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 16 * self.intermediate_shape)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x
