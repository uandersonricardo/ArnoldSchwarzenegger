from argparse import Namespace

from tianshou.data import ReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy
from torch.nn import Module

from src.levd.algorithm.dqn import DQNImpl
from src.levd.network import DuelingDQN


class DuelingDQNImpl(DQNImpl):
    def __init__(self, args: Namespace, log_path: str):
        super(DuelingDQNImpl, self).__init__(args, log_path)

    def init_network(self) -> Module:
        return DuelingDQN(
            self.args.state_shape,
            self.args.action_shape,
            device=self.args.device
        )
