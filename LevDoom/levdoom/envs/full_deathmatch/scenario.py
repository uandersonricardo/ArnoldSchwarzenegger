from collections import deque
from typing import List, Dict

import numpy as np

from levdoom.envs.base import DoomEnv
from levdoom.utils.utils import distance_traversed
from levdoom.utils.wrappers import WrapperHolder, GameVariableRewardWrapper, MovementRewardWrapper

default_reward_values = {
    'BASE_REWARD': 0.,
    'DISTANCE': 0.,
    'KILL': 5.,
    'DEATH': -5.,
    'SUICIDE': -5.,
    'MEDIKIT': 1.,
    'ARMOR': 1.,
    'INJURED': -1.,
    'WEAPON': 1.,
    'AMMO': 1.,
    'USE_AMMO': -0.2,
}

class FullDeathmatch(DoomEnv):
    """
    In this scenario, the agent is randomly spawned in one of 20 possible locations within a maze-like environment, and
    equipped with a weapon and unlimited ammunition. A fixed number of enemies are spawned at random locations at the
    beginning of an episode. Additional enemies will continually be added at random unoccupied locations after a time
    interval. The enemies are rendered immobile, forcing them to remain at their fixed locations. The goal of the agent
    is to locate and shoot the enemies. The agent can move forward, turn left and right, and shoot. The agent is granted
    a reward for each enemy killed.
    """

    def __init__(self,
                 env: str,
                 reward_kill: float = 1.0,
                 penalty_health_loss: float = -0.01,
                 penalty_ammo_used: float = -0.01,
                 traversal_reward_scaler: float = 0.001,
                 reward_frame_survived: float = 0.01,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.reward_kill = reward_kill
        self.penalty_health_loss = penalty_health_loss
        self.penalty_ammo_used = penalty_ammo_used
        self.traversal_reward_scaler = traversal_reward_scaler
        self.reward_frame_survived = reward_frame_survived

        self.kills = 0
        self.deaths = 0
        self.suicides = 0
        self.frags = 0
        self.medikits = 0
        self.armors = 0
        self.pistol = 0
        self.shotgun = 0
        self.chaingun = 0
        self.rocket_launcher = 0
        self.plasma_rifle = 0
        self.bfg9000 = 0
        self.bullets = 0
        self.shells = 0
        self.rockets = 0
        self.cells = 0

        self.distance_buffer = []
        self.ammo_used = 0
        self.ammo_found = 0
        self.hits_taken = 0
        self.frames_survived = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        '''
        game_var_buf:
            [0]  HEALTH
            [1]  KILLCOUNT
            [2]  AMMO2 (Bullets)
            [3]  POSITION_X
            [4]  POSITION_Y
            [5]  FRAGCOUNT
            [6]  SELECTED_WEAPON
            [7]  ARMOR
            [8]  DEAD
            [9]  AMMO3 (Shells)
            [10] AMMO5 (Rockets)
            [11] AMMO6 (Cells)
            [12] WEAPON1 (Fist)
            [13] WEAPON2 (Pistol)
            [14] WEAPON3 (Shotgun)
            [15] WEAPON4 (Chaingun)
            [16] WEAPON5 (Rocket Launcher)
            [17] WEAPON6 (Plasma Rifle)
            [18] WEAPON7 (BFG9000)
        '''
        self.frames_survived += 1
        if len(game_var_buf) < 2:
            return

        distance = distance_traversed(game_var_buf, 3, 4)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]

        # Print state
        # self.print_state(current_vars)

        # Kill
        if current_vars[1] > previous_vars[1]:
            self.kills += current_vars[1] - previous_vars[1]

        # Death
        if current_vars[8] > previous_vars[8]:
            self.deaths += 1
        
        # Suicide
        if current_vars[5] < previous_vars[5]:
            self.suicides += 1

        # Found/lost health
        if current_vars[0] > previous_vars[0]:
            self.medikits += 1
        elif current_vars[0] < previous_vars[0]:
            self.hits_taken += 1
        
        # Found armor
        if current_vars[7] > previous_vars[7]:
            self.armors += 1

        # Changed weapons
        # if current_vars[6] != previous_vars[6]:
        #     print(f' Weapon changed from {previous_vars[6]} to {current_vars[6]}')
        
        # Found weapons
        if current_vars[13] > previous_vars[13]:
            self.pistol += 1
        if current_vars[14] > previous_vars[14]:
            self.shotgun += 1
        if current_vars[15] > previous_vars[15]:
            self.chaingun += 1
        if current_vars[16] > previous_vars[16]:
            self.rocket_launcher += 1
        if current_vars[17] > previous_vars[17]:
            self.plasma_rifle += 1
        if current_vars[18] > previous_vars[18]:
            self.bfg9000 += 1

        # Used ammo
        if current_vars[2] < previous_vars[2] or current_vars[9] < previous_vars[9] or \
           current_vars[10] < previous_vars[10] or current_vars[11] < previous_vars[11]:
            self.ammo_used += 1
        elif current_vars[2] > previous_vars[2] or current_vars[9] > previous_vars[9] or \
           current_vars[10] > previous_vars[10] or current_vars[11] > previous_vars[11]:
            self.ammo_found += 1

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        attack = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward:
                for a in attack:
                    actions.append(t + m + a)
        return actions

    def reward_wrappers_easy(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(MovementRewardWrapper, scaler=self.traversal_reward_scaler),
            WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=1),
            WrapperHolder(GameVariableRewardWrapper, reward=self.penalty_health_loss, var_index=2, decrease=True),
            WrapperHolder(GameVariableRewardWrapper, reward=self.penalty_ammo_used, var_index=3, decrease=True),
        ]

    def reward_wrappers_hard(self) -> List[WrapperHolder]:
        return [WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=0)]

    def extra_statistics(self) -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}
        variables = self.game_variable_buffer[-1]
        return {'health': variables[0],
                'kills': variables[1],
                'ammo_left': variables[2],
                'movement': np.mean(self.distance_buffer).round(3),
                'hits_taken': self.hits_taken}

    def clear_episode_statistics(self):
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.ammo_used = 0
        self.ammo_found = 0
        self.hits_taken = 0
        self.frames_survived = 0
        self.kills = 0
        self.deaths = 0
        self.suicides = 0
        self.frags = 0
        self.medikits = 0
        self.armors = 0
        self.pistol = 0
        self.shotgun = 0
        self.chaingun = 0
        self.rocket_launcher = 0
        self.plasma_rifle = 0
        self.bfg9000 = 0
        self.bullets = 0
        self.shells = 0
        self.rockets = 0
        self.cells = 0
    
    def print_state(self, vars: List[float]) -> None:
        print(f' Health: {vars[0]} | Kills: {vars[1]} | Ammo: {vars[2]} | Position: ({vars[3]:.2f}, {vars[4]:.2f}) | Frags: {vars[5]} | Selected Weapon: {vars[6]} | Armor: {vars[7]} | Dead: {vars[8]} | Bullets: {vars[2]} | Shells: {vars[9]} | Rockets: {vars[10]} | Cells: {vars[11]}')
