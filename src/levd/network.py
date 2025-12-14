from typing import Any, Dict, Optional, Sequence, Tuple, Union
from pathlib import Path

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

        # tf_model = BaseConvNet(
        #     state_shape=state_shape,
        #     labels_shape=4,
        #     device=device,
        # )

        # CHECKPOINT_DIR = Path("tf_checkpoints")
        # checkpoint = torch.load(CHECKPOINT_DIR / "best_map01_checkpoint.pth", map_location=device)
        # tf_model.load_state_dict(checkpoint['model_state_dict'])
        # print(tf_model.conv_head)
        # self.conv_head = tf_model.conv_head

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
                nn.Linear(self.output_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, np.prod(action_shape))
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
    ) -> None:
        super().__init__(state_shape, action_shape, device, features_only=True)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

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

        if state is None:
            state = {}
            h_n = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device)
            c_n = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device)
        else:
            h_n = state["h"]
            c_n = state["c"]

        out, (h_n, c_n) = self.rnn(features, (h_n, c_n))

        q = self.head(out.contiguous().view(batch_size * seq_len, -1))

        state["h"] = h_n
        state["c"] = c_n
        
        return q, state

class DTQN(DQN):
    """Deep Transformer Q-Network.
    Uses a Transformer encoder over extracted visual features.
    """

    def __init__(
        self,
        state_shape: Box,
        action_shape: Discrete,
        device: Union[str, int, torch.device] = "cpu",
        hidden_size: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
    ) -> None:
        # Extract features only from conv head
        super().__init__(state_shape, action_shape, device, features_only=True)

        # Project feature dimension to transformer hidden size
        self.input_proj = nn.Linear(self.output_dim, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head maps the encoded representation to Q-values
        self.head = nn.Linear(hidden_size, np.prod(action_shape))
        self.output_dim = np.prod(action_shape)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, *).
        Expects single-step inputs shaped like images; creates a sequence length 1 internally.
        """
        # Extract per-step features using conv head
        features, _ = super().forward(obs)

        # Create sequence length = 1 for transformer (B, S=1, D)
        x = features.unsqueeze(1)
        x = self.input_proj(x)

        # For single-step, no mask is required; pass through encoder
        # Output shape: (B, S=1, hidden_size)
        enc = self.encoder(x)

        # Use the last token (here the only token)
        q = self.head(enc[:, -1, :])

        return q, state

class DuelingDQN(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape: Box,
        action_shape: Discrete,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__(state_shape, action_shape, device, features_only=True)
        self.action_num = np.prod(action_shape)

        self.Q = nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.action_num)
        )
        self.V = nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        self.output_dim = self.action_num

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        # print(info)
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, 1)
        v = self.V(obs)
        v = v.view(-1, 1, 1)
        logits = q - q.mean(dim=1, keepdim=True) + v
        # probs = logits.softmax(dim=2)
        return probs, state


class BaseConvNet(nn.Module):
    """A base class for convolutional neural networks used in RL."""

    def __init__(
        self,
        state_shape: Box,
        labels_shape: int,
        device: Union[str, int, torch.device] = "cpu"):
        super(BaseConvNet, self).__init__()
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

        self.net = nn.Sequential(
            self.net,
            nn.Linear(self.output_dim, labels_shape),
            nn.Sigmoid()
        )
        self.output_dim = labels_shape

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs)


def build_conv_head(in_channels: int):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)
