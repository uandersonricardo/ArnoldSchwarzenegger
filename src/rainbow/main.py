import os

import numpy as np
import torch

from src.doom.actions import ActionBuilder
from src.doom.game import Game
from src.doom.reward import parse_reward_values
from src.logger import get_logger
from src.rainbow.args import parse_game_args
from src.rainbow.dqn import DQNAgent
from src.utils import get_dump_path

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

# arguments parsing

# Seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.cudnn.enabled:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Parse arguments
args = parse_game_args()

dump_path = get_dump_path(args.main_dump_path, args.exp_name)
logger = get_logger(filepath=os.path.join(dump_path, 'train.log'))
logger.info('========== Running DOOM ==========')
logger.info('Experiment will be saved in: %s', dump_path)

# Initialize the game
action_builder = ActionBuilder(args)

game = Game(
    scenario=args.wad,
    action_builder=action_builder,
    reward_values=parse_reward_values(args.reward_values),
    score_variable='USER2',
    freedoom=args.freedoom,
    screen_resolution='RES_400X225',
    use_screen_buffer=args.use_screen_buffer,
    use_depth_buffer=args.use_depth_buffer,
    labels_mapping=args.labels_mapping,
    game_features=args.game_features,
    mode=('SPECTATOR' if args.human_player else 'PLAYER'),
    player_rank=args.player_rank,
    players_per_game=args.players_per_game,
    render_hud=args.render_hud,
    render_crosshair=args.render_crosshair,
    render_weapon=args.render_weapon,
    freelook=args.freelook,
    visible=args.visualize,
    n_bots=args.n_bots,
    use_scripted_marines=True
)

# Initialize the DQN agent
agent = DQNAgent(game, memory_size, batch_size, target_update, epsilon_decay, seed)

# Train
agent.train(num_frames)

# Test
agent.test()
