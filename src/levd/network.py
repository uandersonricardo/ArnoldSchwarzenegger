from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn

from tianshou.utils.net.discrete import NoisyLinear


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape: Box,
        action_shape: Discrete,
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.conv_head = build_conv_head(state_shape[0])
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            self.conv_head,
            self.flatten,
        )
        # Calculate output_dim correctly
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_shape)
            self.output_dim = self.flatten(self.conv_head(dummy_input)).shape[1]

        if not features_only:
            self.net = nn.Sequential(
                self.net,
                nn.Linear(self.output_dim, np.prod(action_shape))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net,
                nn.Linear(self.output_dim, output_dim),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


class C51(DQN):
    """Reference: A distributional perspective on reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape: Box,
        action_shape: Discrete,
        num_atoms: int = 51,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(state_shape, [self.action_num * num_atoms], device)
        self.num_atoms = num_atoms

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.num_atoms).softmax(dim=-1)
        obs = obs.view(-1, self.action_num, self.num_atoms)
        return obs, state


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape: Box,
        action_shape: Discrete,
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        device: Union[str, int, torch.device] = "cpu",
        is_dueling: bool = True,
        is_noisy: bool = True,
        n_features: int = 0,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__(state_shape, action_shape, device, features_only=True)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms
        self.n_features = n_features

        def linear(x, y):
            return NoisyLinear(x, y, noisy_std) if is_noisy else nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512), nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms)
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512), nn.ReLU(inplace=True),
                linear(512, self.num_atoms)
            )
        
        # Game features prediction branch (auxiliary task)
        if self.n_features > 0:
            self.proj_game_features = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, self.n_features),
                nn.Sigmoid()  # Binary predictions
            )
        
        self.output_dim = self.action_num * self.num_atoms

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        
        # Game features prediction
        if self.n_features > 0:
            features = self.proj_game_features(obs)
            return (probs, features), state
        
        return probs, state


class DRQN(DQN):
    """Deep Recurrent Q-Network.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape: Box,
        action_shape: Discrete,
        device: Union[str, int, torch.device] = "cpu",
        hidden_size: int = None,
        num_layers: int = 1,
    ) -> None:
        super().__init__(state_shape, action_shape, device, features_only=True)

        self.rnn = nn.LSTM(
            input_size=self.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, np.prod(action_shape))
        self.output_dim = np.prod(action_shape)
        self.h_n = None
        self.c_n = None

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        features, _ = super().forward(obs)

        features = features.unsqueeze(1)
        batch_size, seq_len, feat_dim = features.shape

        if self.h_n is None or self.h_n.size(1) != batch_size:
            h_n = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device)
        else:
            h_n = self.h_n

        if self.c_n is None or self.c_n.size(1) != batch_size:
            c_n = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device)
        else:
            c_n = self.c_n

        out, (h_n, c_n) = self.rnn(features, (h_n, c_n))

        self.h_n = h_n
        self.c_n = c_n

        q = self.head(out.contiguous().view(batch_size * seq_len, -1))
        
        return q, state


def build_conv_head(in_channels: int):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(p=0.0),
    ]

    return nn.Sequential(*layers)
