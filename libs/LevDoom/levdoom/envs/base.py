import os
import time
from collections import deque
from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import cv2
import gymnasium
import numpy as np
import vizdoom as vzd
from vizdoom import ScreenResolution, GameVariable

from levdoom.utils.utils import get_screen_resolution


class DoomEnv(gymnasium.Env):

    def __init__(self,
                 env: str = 'default',
                 frame_skip: int = 4,
                 seed: int = 0,
                 render: bool = True,
                 resolution: str = None,
                 max_steps: int = None,
                 variable_queue_length: int = 5,
                 n_bots: int = 10,
                 use_labels: bool = False,
                 game_features: List[str] = None):
        super().__init__()
        self.env = env
        self.frame_skip = frame_skip
        self.scenario = self.__module__.split('.')[-2]
        self.n_bots = n_bots
        
        # Game features configuration
        self.use_labels = use_labels
        self.game_features = game_features or []

        # Determine the directory of the doom scenario
        scenario_dir = f'{Path(__file__).parent.resolve()}/{self.scenario}'

        # Create the Doom game instance
        self.game = vzd.DoomGame()
        self.game.load_config(f"{scenario_dir}/conf.cfg")
        self.game.set_doom_scenario_path(f"{scenario_dir}/maps/{env}.wad")
        self.game.set_seed(seed)
        self.render_enabled = render

        if max_steps:
            self.game.set_episode_timeout(max_steps)
        if render:
            # Use a higher resolution for rendering gameplay
            # self.game.set_screen_resolution(ScreenResolution.RES_400X225)
            # self.frame_skip = 1
            pass
        elif resolution:  # Use a particular predefined resolution
            self.game.set_screen_resolution(get_screen_resolution(resolution))

        self.game.add_game_args(
            "-host 1 -deathmatch +sv_cheats 1 "
            "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
            " +name AI +colorset 0 "
        )
        
        # Enable labels buffer for game features
        if self.use_labels or self.game_features:
            self.game.set_labels_buffer_enabled(True)

        self.game.init()
        
        self.update_bots()

        # Define the observation space
        self.game_res = (self.game.get_screen_height(), self.game.get_screen_width(), 3)
        self._observation_space = gymnasium.spaces.Box(low=0, high=255, shape=self.game_res, dtype=np.uint8)

        # Define the action space
        self.available_actions = self.get_available_actions()
        self._action_space = gymnasium.spaces.Discrete(len(self.available_actions))

        # Initialize and fill the game variable queue
        self.game_variable_buffer = deque(maxlen=variable_queue_length)
        self.game_variable_buffer.append(self.game.get_state().game_variables)

        # self.extra_statistics = ['kills', 'health', 'ammo', 'movement', 'kits_obtained', 'hits_taken']

    @property
    def name(self) -> str:
        return f'{self.scenario}-{self.env}'

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return self._action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return self._observation_space

    def extra_statistics(self) -> Dict[str, float]:
        """
        Retrieves additional statistics specific to the scenario. Mostly game variables.

        Args:
            mode (str): A specifier to distinguish which environment the statistic is for (train/test).

        Returns:
            statistics (Dict[str, float]): A dictionary containing additional statistical data.
        """
        return {}

    def store_statistics(self, game_vars: deque) -> None:
        """
        Stores statistics based on the game variables.

        Args:
            game_vars (deque): A deque containing game variables for statistics.
        """
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state and returns the initial observation.

        Args:
            seed (Optional[int]): Seed for random number generator.
            options (Optional[dict]): Additional options for environment reset.

        Returns:
            observation (np.ndarray): Initial state observation of the environment.
            info (Dict[str, Any]): Additional information about the initial state.
        """
        try:
            self.game.new_episode()
        except vzd.ViZDoomIsNotRunningException:
            print('ViZDoom is not running. Restarting...')
            self.game.init()
            self.game.new_episode()
        
        self.update_bots()

        self.clear_episode_statistics()
        state = self.game.get_state().screen_buffer
        state = np.transpose(state, [1, 2, 0])
        self.obs_dtype = state.dtype
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform an action in the environment and observe the result.

        Args:
            action (int): An action provided by the agent.

        Returns:
            observation (np.ndarray): The current state observation after taking the action.
            reward (float): The reward achieved by the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (Dict[str, Any]): Additional information about the environment and episode.
        """
        action = self.available_actions[action]
        self.game.set_action(action)
        self.game.advance_action(self.frame_skip)

        state = self.game.get_state()
        reward = 0.0
        done = self.game.is_player_dead() or not state
        truncated = self.game.is_episode_finished()
        info = self.extra_statistics()

        observation = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.zeros(self.game_res, dtype=self.obs_dtype)
        if not done:
            self.game_variable_buffer.append(state.game_variables)
        
        # Extract game features
        if self.game_features and state:
            info['game_features'] = self._extract_game_features(state)

        self.store_statistics(self.game_variable_buffer)
        return observation, reward, done, truncated, info

    def get_available_actions(self) -> List[List[float]]:
        raise NotImplementedError

    def reward_wrappers_easy(self) -> List[gymnasium.RewardWrapper]:
        """
        Returns a list of reward wrapper classes for the dense reward setting.

        Returns:
            List[gymnasium.RewardWrapper]: A list of reward wrapper classes.
        """
        raise NotImplementedError

    def reward_wrappers_hard(self) -> List[gymnasium.RewardWrapper]:
        """
        Returns a list of reward wrapper classes for the sparse reward setting.

        Returns:
            List[gymnasium.RewardWrapper]: A list of reward wrapper classes.
        """
        raise NotImplementedError

    def get_and_update_user_var(self, game_var: GameVariable) -> int:
        """
        Retrieves and updates a user-defined variable from the game.

        Args:
            game_var (GameVariable): The game variable to retrieve and update.

        Returns:
            prev_var (int): The previous value of the specified game variable.
        """
        prev_var = self.user_variables[game_var]
        self.user_variables[game_var] = self.game.get_game_variable(game_var)
        return prev_var

    def render(self, mode="human"):
        """
        Renders the current state of the environment based on the specified mode.

        Args:
            mode (str): The mode for rendering (e.g., 'human', 'rgb_array').

        Returns:
            img (List[np.ndarray] or np.ndarray): Rendered image of the environment state.
        """
        state = self.game.get_state()
        img = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.uint8(np.zeros(self.game_res))
        if mode == 'human':
            if not self.render_enabled:
                return [img]
            try:
                # Render the image to the screen with swapped red and blue channels
                cv2.imshow('DOOM', img[:, :, [2, 1, 0]])
                cv2.waitKey(1)
                # time.sleep(0.02)
            except Exception as e:
                print(f'Screen rendering unsuccessful: {e}')
                return np.zeros(img.shape)
        return [img]

    def clear_episode_statistics(self) -> None:
        """
        Clears or resets statistics collected during an episode.
        """
        self.game_variable_buffer.clear()

    def close(self):
        """
        Closes the Doom game instance.
        """
        self.game.close()
    
    def update_bots(self):
        """
        Updates the bots in the game.
        """
        self.game.send_game_command("removebots")

        for _ in range(10):
            self.game.send_game_command("addbot")
    
    def _extract_game_features(self, state) -> Optional[np.ndarray]:
        """
        Extract game features from labels buffer.
        Returns a binary vector indicating presence of each feature type.
        """
        if not self.game_features or not state.labels_buffer is None:
            return None
        
        try:
            labels = state.labels
            if not labels:
                return np.zeros(len(self.game_features), dtype=np.float32)
            
            # Define object sets (similar to Arnold's labels.py)
            ENEMY_SET = {
                'MarineBFG', 'MarineBerserk', 'MarineChaingun', 'MarineChainsaw',
                'MarineFist', 'MarinePistol', 'MarinePlasma', 'MarineRailgun',
                'MarineRocket', 'MarineSSG', 'MarineShotgun', 'Demon',
                'Zombieman', 'ShotgunGuy', 'ChaingunGuy', 'Imp', 'Cacodemon',
                'HellKnight', 'BaronOfHell', 'Archvile', 'Revenant', 'Mancubus',
                'Arachnotron', 'PainElemental', 'Fatso'
            }
            HEALTH_SET = {
                'ArmorBonus', 'BlueArmor', 'GreenArmor', 'HealthBonus',
                'Medikit', 'Stimpack', 'Soulsphere', 'Megasphere'
            }
            WEAPON_SET = {
                'Pistol', 'Chaingun', 'RocketLauncher', 'Shotgun', 'SuperShotgun',
                'PlasmaRifle', 'BFG9000', 'Chainsaw'
            }
            AMMO_SET = {
                'Cell', 'CellPack', 'Clip', 'ClipBox', 'RocketAmmo', 'RocketBox',
                'Shell', 'ShellBox', 'Backpack'
            }
            
            feature_sets = {
                'enemy': ENEMY_SET,
                'health': HEALTH_SET,
                'weapon': WEAPON_SET,
                'ammo': AMMO_SET
            }
            
            # Check if any object of each type is visible
            features = []
            for feature_name in self.game_features:
                if feature_name in feature_sets:
                    visible = any(label.object_name in feature_sets[feature_name] 
                                 for label in labels)
                    features.append(float(visible))
                else:
                    features.append(0.0)
            
            return np.array(features, dtype=np.float32) if features else None
        except Exception as e:
            # Return zeros on error to avoid breaking training
            return np.zeros(len(self.game_features), dtype=np.float32)
