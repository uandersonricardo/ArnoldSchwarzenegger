import os
from time import sleep

import vizdoom as vzd


game = vzd.DoomGame()

# Use your config
game.load_config(os.path.join(vzd.scenarios_path, "cig.cfg"))
# game.set_doom_map("map01")
wad_path = "/home/uanderson/mestrado/deep-learning/Arnold/LevDoom/levdoom/envs/full_deathmatch/maps/full_deathmatch.wad"
game.set_doom_scenario_path(wad_path)

game.set_doom_map("map01")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# Start multiplayer game only with your AI
# (with options that will be used in the competition, details in cig_mutliplayer_host.py example).
game.add_game_args(
    "-host 1 -deathmatch +timelimit 10.0 "
    "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
    "+viz_respawn_delay 10 +viz_nocheat 1"
)

# Join existing game.
# game.add_game_args("-join 127.0.0.1")  # Connect to a host for a multiplayer game.

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name AI +colorset 0")
game.set_mode(vzd.Mode.PLAYER)
game.set_console_enabled(True)

# game.set_window_visible(False)

# game.init()

game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_window_visible(True)
# game.set_mode(Mode.PLAYER)
game.set_mode(vzd.Mode.ASYNC_PLAYER)

# Wait for wad generator
# sleep(1)
# while not os.path.exists(wad_path):
#     sleep(1)

game.init()
i = 1
# Play until the game is over.
while True:
    print(f"Map {i}")
    while not game.is_episode_finished():
        state = game.get_state()
        assert state is not None

        game.advance_action()
        last_action = game.get_last_action()
        reward = game.get_last_reward()

        if game.is_player_dead():
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

        print(f"State #{state.number}")
        print("Game variables: ", state.game_variables)
        print("Action:", last_action)
        print("Reward:", reward)
        print("=====================")

    print("Episode finished!")
    print("Total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

    game.new_episode()
    i += 1

game.close()
