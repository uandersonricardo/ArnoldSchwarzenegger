from argparse import Namespace
from typing import Dict, Union

import torch
from tianshou.data import ReplayBuffer, Collector, VectorReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import BaseLogger
from torch.nn import Module
from torch.optim import Optimizer

from src.levd.algorithm.base import BaseImpl
from src.levd.algorithm.dqn import DQNImpl
from src.levd.network import DTQN


class DTQNImpl(DQNImpl):
    def __init__(self, args: Namespace, log_path: str):
        super(DTQNImpl, self).__init__(args, log_path)

    def create_trainer(self, train_collector: Collector, test_collector: Collector, logger: BaseLogger) -> Dict[
        str, Union[float, str]]:
        return offpolicy_trainer(
            self.policy,
            train_collector,
            test_collector,
            self.args.epoch,
            self.args.step_per_epoch,
            self.args.step_per_collect,
            self.args.test_num,
            self.args.batch_size,
            train_fn=self.train_fn,
            test_fn=self.test_fn,
            save_best_fn=self.save_best_fn,
            save_checkpoint_fn=self.save_checkpoint_fn,
            logger=logger,
            update_per_step=self.args.update_per_step,
            test_in_train=False
        )

    def init_network(self) -> Module:
        return DTQN(self.args.state_shape, self.args.action_shape, hidden_size=self.args.hidden_size, device=self.args.device)

    def init_policy(self) -> BasePolicy:
        return DQNPolicy(
            self.network,
            self.optimizer,
            self.args.gamma,
            self.args.n_step,
            target_update_freq=self.args.target_update_freq
        ).to(self.args.device)
