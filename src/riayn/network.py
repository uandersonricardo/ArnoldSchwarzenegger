import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, in_dim: tuple[int], out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim[0] * in_dim[1] * in_dim[2], 128),
            # nn.Linear(in_dim[0] * in_dim[1] * in_dim[2], 128),Z
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        if len(x.shape) > 3:
            flattened = x.view(x.size(0), -1)
            return self.layers(flattened)
        else:
            flattened = x.view(-1)
            return self.layers(flattened)
