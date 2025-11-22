import pathlib
import pprint
from argparse import Namespace
from datetime import datetime
from functools import partial

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tianshou.data.collector import Collector
from tianshou.env import ShmemVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.utils.logger.wandb import WandbLogger

import levdoom
from levdoom import Scenario
from src.levd.config import parse_args, Algorithm


# Watch the agent's performance
def watch():
    print("Setup test envs ...")
    policy.eval()
    test_envs.seed(args.seed)
    if args.save_buffer_name:
        print(f"Generate buffer with size {args.buffer_size}")
        buffer = algorithm.create_buffer(len(test_envs))
        extra_statistics = ['kills', 'health', 'ammo', 'movement', 'kits_obtained', 'hits_taken']
        collector = Collector(policy, test_envs, buffer, exploration_noise=True, extra_statistics=extra_statistics)
        result = collector.collect(n_step=args.buffer_size, frame_skip=args.frame_skip)
        print(f"Save buffer into {args.save_buffer_name}")
        # Unfortunately, pickle will cause oom with 1M buffer size
        buffer.save_hdf5(args.save_buffer_name)
    else:
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render_sleep,
                                        frame_skip=args.frame_skip)
    reward = result["reward"].mean()
    lengths = result["length"].mean() * args.frame_skip
    print(f'Mean reward (over {result["n/ep"]} episodes): {reward}')
    print(f'Mean length (over {result["n/ep"]} episodes): {lengths}')


if __name__ == '__main__':
    watch(parse_args())
