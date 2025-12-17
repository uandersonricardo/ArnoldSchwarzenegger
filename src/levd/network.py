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
        n_channels, height, width = state_shape
        channels = 3  # RGB
        n_stack = n_channels // channels
        temp_state_shape = (state_shape[0] // n_stack, state_shape[1], state_shape[2])

        super().__init__(temp_state_shape, action_shape, device, features_only=True)

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
        # Reshape obs from (batch_size, n_channels, height, width)
        # to (batch_size * n_stack, channels, height, width)
        batch_size, n_channels, height, width = obs.shape
        channels = 3  # RGB
        n_stack = n_channels // channels
        obs = obs.reshape(batch_size, n_stack, channels, height, width)

        # Now reshape to (batch_size * n_stack, channels, height, width)
        obs = obs.reshape(batch_size * n_stack, channels, height, width)

        # Get features from CNN
        features, state = super().forward(obs)

        # Reshape features back to (batch_size, n_stack, feature_dim)
        features = features.view(batch_size, n_stack, -1)

        # Pass through RNN
        batch_size, seq_len, feat_dim = features.shape

        if state is None:
            state = {}
            h_n = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device)
            c_n = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device)
        else:
            h_n = state["hidden"].transpose(0, 1).contiguous()
            c_n = state["cell"].transpose(0, 1).contiguous()

        out, (h_n, c_n) = self.rnn(features, (h_n, c_n))

        q = self.head(out.contiguous().view(batch_size * seq_len, -1))
        q = q.view(batch_size, seq_len, -1)

        # Drop the sequence dimension for Q-values
        q = q[:, -1, :]

        state["hidden"] = h_n.transpose(0, 1).contiguous()
        state["cell"] = c_n.transpose(0, 1).contiguous()

        return q, state

class DTQN(DQN):
    """Deep Transformer Q-Network.
    
    Uses transformer architecture instead of LSTM for temporal processing.
    Suitable for learning from sequences of observations.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape: Box,
        action_shape: Discrete,
        device: Union[str, int, torch.device] = "cpu",
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        context_length: int = 8,
    ) -> None:
        n_channels, height, width = state_shape
        channels = 3  # RGB
        n_stack = n_channels // channels
        temp_state_shape = (state_shape[0] // n_stack, state_shape[1], state_shape[2])

        super().__init__(temp_state_shape, action_shape, device, features_only=True)
        
        self.hidden_size = hidden_size
        self.context_length = context_length
        
        # Project feature dimension to hidden_size
        self.feature_proj = nn.Linear(self.output_dim, hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, context_length, hidden_size),
            requires_grad=True
        )
        
        # Decoder with causal masking - only sees past
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # For decoder, we need a memory (can use zeros or learned)
        self.memory = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Output head
        self.head = nn.Linear(hidden_size, np.prod(action_shape))
        self.output_dim = np.prod(action_shape)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        batch_size, n_channels, height, width = obs.shape
        channels = 3  # RGB
        n_stack = n_channels // channels
        obs = obs.reshape(batch_size, n_stack, channels, height, width)

        # Now reshape to (batch_size * n_stack, channels, height, width)
        obs = obs.reshape(batch_size * n_stack, channels, height, width)

        # Get features from CNN
        features, state = super().forward(obs)
        
        # Project features to hidden dimension
        features = self.feature_proj(features)
        
        # Reshape features back to (batch_size, n_stack, hidden_size)
        features = features.view(batch_size, n_stack, -1)
        
        # Pass through Transformer
        batch_size, seq_len, feat_dim = features.shape
        
        # Add positional encoding
        features = features + self.pos_encoding[:, :seq_len, :]


        # Decoder requires causal mask and memory
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=self.device
        )

        memory = self.memory.expand(batch_size, -1, -1)

        transformer_out = self.transformer(
            features, 
            memory,
            tgt_mask=causal_mask
        )
        
        # Take the last output for Q-value prediction
        last_output = transformer_out[:, -1, :]
        
        # Compute Q-values
        q = self.head(last_output)
        
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
