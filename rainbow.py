import argparse
import random
from typing import SupportsFloat

import cv2
import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from vizdoom import gymnasium_wrapper # pylint: disable=unused-import

from src.rainbow.replay_buffer import *
from src.rainbow.dqn import DQN

# Constants
DEFAULT_ENV = "VizdoomDeathmatch-v1"
IMAGE_SHAPE = (108, 60)
FRAME_SKIP = 4

class ObservationWrapper(gym.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.
    """

    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]

        # Create new observation space with the new shape
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)

        self.observation_space = gym.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        return cv2.resize(observation["screen"], self.image_shape_reverse)


class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        self.env = self.wrap_env(gym.make(env_name, render_mode="human", frame_skip=FRAME_SKIP))
        self.env_evaluate = self.wrap_env(gym.make(env_name, render_mode="human", frame_skip=FRAME_SKIP))  # When evaluating the policy, we need to rebuild an environment

        self.env.action_space.seed(seed)
        self.env_evaluate.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.args.state_dim = self.env.observation_space.shape
        self.args.action_dim = self.env.action_space["binary"].n
        self.args.episode_limit = self.args.episode_limit  # Maximum number of steps per episode
        print("env={}".format(self.env_name))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        if args.use_per and args.use_n_steps:
            self.replay_buffer = NStepsPrioritizedReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = NStepsReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_Steps"

        self.writer = SummaryWriter(log_dir='runs/DQN/{}_env_{}_number_{}_seed_{}'.format(self.algorithm, env_name, number, seed))

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self, ):
        self.evaluate_policy()
        while self.total_steps < self.args.max_train_steps:
            state, _ = self.env.reset()
            done = False
            episode_steps = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step({"binary": action, "continuous": [0.0, 0.0, 0.0] })
                done = terminated or truncated
                episode_steps += 1
                self.total_steps += 1

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # terminal means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.args.episode_limit:
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store(state, action, reward, next_state, terminal, done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.evaluate_policy()
        # Save reward
        np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.algorithm, self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))

    def evaluate_policy(self, ):
        evaluate_reward = 0
        self.agent.net.eval()
        for _ in range(self.args.evaluate_times):
            state, _ = self.env_evaluate.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=0)
                next_state, reward, terminated, truncated, _ = self.env_evaluate.step({"binary": action, "continuous": [0.0, 0.0, 0.0] })
                done = terminated or truncated
                episode_reward += reward
                state = next_state
            evaluate_reward += episode_reward
        self.agent.net.train()
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
        self.writer.add_scalar('step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)

    def wrap_env(self, env):
        # Customize the reward function
        env.unwrapped.game.set_living_reward(-0.01)
        env.unwrapped.game.set_death_reward(-1.0)
        env.unwrapped.game.set_map_exit_reward(10.0)
        env.unwrapped.game.set_kill_reward(1.0)
        env.unwrapped.game.set_frag_reward(1.0)
        env.unwrapped.game.set_secret_reward(1.0)
        env.unwrapped.game.set_item_reward(0.5)
        env.unwrapped.game.set_damage_made_reward(0.1)
        env.unwrapped.game.set_hit_reward(0.1)
        env.unwrapped.game.set_hit_taken_reward(-0.1)
        env.unwrapped.game.set_damage_taken_reward(-0.1)

        # Apply observation wrapper and reward scaling
        env = ObservationWrapper(env)
        env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--episode_limit", type=int, default=int(1e6), help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")

    parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()

    runner = Runner(args=args, env_name=DEFAULT_ENV, number=1, seed=args.seed)
    runner.run()
