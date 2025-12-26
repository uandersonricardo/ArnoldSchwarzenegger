from argparse import Namespace

from tianshou.data import ReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import BasePolicy, RainbowPolicy, C51Policy
from torch.nn import Module

from src.levd.algorithm.dqn import DQNImpl
from src.levd.network import Rainbow, C51
    

class C51Impl(DQNImpl):
    def __init__(self, args: Namespace, log_path: str):
        super(C51Impl, self).__init__(args, log_path)

    def init_network(self) -> Module:
        return C51(
            self.args.state_shape,
            self.args.action_shape,
            self.args.num_atoms,
            device=self.args.device
        )

    def init_policy(self) -> BasePolicy:
        return C51Policy(
            self.network,
            self.optimizer,
            self.args.gamma,
            self.args.num_atoms,
            self.args.v_min,
            self.args.v_max,
            self.args.n_step,
            target_update_freq=self.args.target_update_freq
        ).to(self.args.device)
