import argparse

from src.utils import bool_flag, map_ids_flag, bcast_json_list
from src.doom.utils import get_n_feature_maps
from src.doom.game_features import parse_game_features

def parse_game_args():
    parser = argparse.ArgumentParser(description='Arnold Schwarzenegger')

    # Experiment name / dump path
    parser.add_argument("--main_dump_path", type=str, default="./dumped",
                        help="Main dump path")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Experiment name")

    # Doom scenario / map ID
    parser.add_argument("--scenario", type=str, default="deathmatch",
                        help="Doom scenario")
    parser.add_argument("--map_ids_train", type=map_ids_flag, default=map_ids_flag("1"),
                        help="Train map IDs")
    parser.add_argument("--map_ids_test", type=map_ids_flag, default=map_ids_flag("1"),
                        help="Test map IDs")

    # general game options (freedoom, screen resolution, available buffers,
    # game features, things to render, history size, frame skip, etc)
    parser.add_argument("--freedoom", type=bool_flag, default=True,
                        help="Use freedoom2.wad (as opposed to DOOM2.wad)")
    parser.add_argument("--height", type=int, default=60,
                        help="Image height")
    parser.add_argument("--width", type=int, default=108,
                        help="Image width")
    parser.add_argument("--gray", type=bool_flag, default=False,
                        help="Use grayscale")
    parser.add_argument("--use_screen_buffer", type=bool_flag, default=True,
                        help="Use the screen buffer")
    parser.add_argument("--use_depth_buffer", type=bool_flag, default=False,
                        help="Use the depth buffer")
    parser.add_argument("--labels_mapping", type=str, default='',
                        help="Map labels to different feature maps")
    parser.add_argument("--game_features", type=str, default='',
                        help="Game features")
    parser.add_argument("--render_hud", type=bool_flag, default=False,
                        help="Render HUD")
    parser.add_argument("--render_crosshair", type=bool_flag, default=True,
                        help="Render crosshair")
    parser.add_argument("--render_weapon", type=bool_flag, default=True,
                        help="Render weapon")
    parser.add_argument("--hist_size", type=int, default=4,
                        help="History size")
    parser.add_argument("--frame_skip", type=int, default=4,
                        help="Number of frames to skip")

    # Available actions
    # combination of actions the agent is allowed to do.
    # this is for non-continuous mode only, and is ignored in continuous mode
    parser.add_argument("--action_combinations", type=str,
                        default='move_fb+turn_lr+move_lr+attack',
                        help="Allowed combinations of actions")
    # freelook: allow the agent to look up and down
    parser.add_argument("--freelook", type=bool_flag, default=False,
                        help="Enable freelook (look up / look down)")
    # speed and crouch buttons: in non-continuous mode, the network can not
    # have control on these buttons, and they must be set to always 'on' or
    # 'off'. In continuous mode, the network can manually control crouch and
    # speed.
    # manual_control makes the agent turn about (180 degrees turn) if it keeps
    # repeating the same action (if it is stuck in one corner, for instance)
    parser.add_argument("--speed", type=str, default='off',
                        help="Crouch: on / off / manual")
    parser.add_argument("--crouch", type=str, default='off',
                        help="Crouch: on / off / manual")
    parser.add_argument("--manual_control", type=bool_flag, default=False,
                        help="Manual control to avoid action repetitions")

    # number of players / games
    parser.add_argument("--players_per_game", type=int, default=1,
                        help="Number of players per game")
    parser.add_argument("--player_rank", type=int, default=0,
                        help="Player rank")

    # miscellaneous
    parser.add_argument("--dump_path", type=str, default=".",
                        help="Folder to store the models / parameters.")
    parser.add_argument("--visualize", type=bool_flag, default=False,
                        help="Visualize")
    parser.add_argument("--evaluate", type=int, default=0,
                        help="Fast evaluation of the model")
    parser.add_argument("--human_player", type=bool_flag, default=False,
                        help="Human player (SPECTATOR mode)")
    parser.add_argument("--reload", type=str, default="",
                        help="Reload previous model")
    parser.add_argument("--dump_freq", type=int, default=0,
                        help="Dump every X iterations (0 to disable)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--log_frequency", type=int, default=100,
                        help="Log frequency (in seconds)")

    # Scenario specific arguments
    parser.add_argument("--wad", type=str, default="",
                        help="WAD scenario filename")
    parser.add_argument("--n_bots", type=int, default=8,
                        help="Number of ACS bots in the game")
    parser.add_argument("--reward_values", type=str, default="",
                        help="reward_values")
    parser.add_argument("--randomize_textures", type=bool_flag, default=False,
                        help="Randomize textures during training")
    parser.add_argument("--init_bots_health", type=int, default=100,
                        help="Initial bots health during training")

    params = parser.parse_args()

    params.human_player = params.human_player and params.player_rank == 0

    # Game variables / Game features / feature maps
    params.game_variables = [('health', 101), ('sel_ammo', 301)]

    # Finalize args
    params.n_variables = len(params.game_variables)
    params.n_features = sum(parse_game_features(params.game_features))
    params.n_fm = get_n_feature_maps(params)

    params.variable_dim = bcast_json_list(params.variable_dim, params.n_variables)
    params.bucket_size = bcast_json_list(params.bucket_size, params.n_variables)

    if not hasattr(params, 'use_continuous'):
        params.use_continuous = False

    # Training / Evaluation parameters
    params.episode_time = None  # episode maximum duration (in seconds)
    params.eval_freq = 20000    # time (in iterations) between 2 evaluations
    params.eval_time = 900      # evaluation time (in seconds)

    return params
