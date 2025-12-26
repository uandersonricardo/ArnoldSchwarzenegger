import cv2
import numpy as np
import torch
import gymnasium
from vizdoom import gymnasium_wrapper

# from src.arnold.utils import get_dump_path
# from src.arnold.logger import get_logger
# from src.arnold.doom.game import Game
# from src.arnold.doom.reward import parse_reward_values
# from src.arnold.doom.actions import ActionBuilder
# from src.riayn.dqn import DQNAgent
from src.riayn.dueling_network_dqn import DuelingNetworkDQNAgent
# from src.riayn.args import parse_game_args

# Parameters
seed = 777
screen_width = 400
screen_height = 225
screen_channels = 3  # GRAY=1, RGB=3, WITH_DEPTH=4, WITH_LABELS=variable
num_frames = 10000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000

# Constants
DEFAULT_ENV = "VizdoomDeathmatch-v0"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]  # type: ignore

# Height and width of the resized image
IMAGE_SHAPE = (60, 80)

# Training parameters
TRAINING_TIMESTEPS = int(1e6)
N_STEPS = 128
N_ENVS = 8
FRAME_SKIP = 4

class ObservationWrapper(gymnasium.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]

        # Create new observation space with the new shape
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)

        # Get audio observation space if available
        if "audio" in env.observation_space.spaces:
            self.observation_space = gymnasium.spaces.Dict(
                {
                    "screen": gymnasium.spaces.Box(
                        0, 255, shape=new_shape, dtype=np.uint8
                    ),
                    "audio": env.observation_space["audio"],
                }
            )
        else:
            self.observation_space = gymnasium.spaces.Dict(
                {
                    "screen": gymnasium.spaces.Box(
                        0, 255, shape=new_shape, dtype=np.uint8
                    )
                }
            )

    def observation(self, observation):
        if "audio" in self.observation_space.spaces:
            observation = {
                "screen": cv2.resize(observation["screen"], self.image_shape_reverse),
                "audio": observation["audio"],
            }
        else:
            observation = {
                "screen": cv2.resize(observation["screen"], self.image_shape_reverse)
            }
        return observation

# Seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.cudnn.enabled:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# if __name__ == "__main__":
#     env = gymnasium.make(
#         "VizdoomHealthGatheringSupreme-v1", render_mode="human", frame_skip=4
#     )

#     # Rendering random rollouts for ten episodes
#     for _ in range(10):
#         done = False
#         obs, info = env.reset(seed=42)
#         while not done:
#             obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
#             done = terminated or truncated

def main():
    # Create multiple environments: this speeds up training with PPO
    # We apply two wrappers on the environment:
    #  1) The above wrapper that modifies the observations (takes only the image and resizes it)
    #  2) A reward scaling wrapper. Normally the scenarios use large magnitudes for rewards (e.g., 100, -100).
    #     This may lead to unstable learning, and we scale the rewards by 1/100
    def wrap_env(env):
        env = ObservationWrapper(env)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

    env = gymnasium.make(DEFAULT_ENV, render_mode="human", frame_skip=FRAME_SKIP)
    # agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)
    agent = DuelingNetworkDQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)
    agent.train(num_frames)

if __name__ == "__main__":
    main()
