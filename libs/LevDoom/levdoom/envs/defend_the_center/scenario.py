from collections import deque
from typing import List, Dict

import numpy as np
from levdoom.envs.base import DoomEnv
from levdoom.utils.wrappers import WrapperHolder, GameVariableRewardWrapper, LabelRewardWrapper
from vizdoom import GameVariable


class DefendTheCenter(DoomEnv):
    """
    In this scenario, the agent is spawned in the center of a circular room. Enemies are spawned at fixed positions
    alongside the wall of the area. The enemies do not possess a projectile attack and therefore have to make their way
    within melee range to inflict damage. The agent is rendered immobile, but equipped with a weapon and limited
    ammunition to fend off the encroaching enemies. Once the enemies are killed, they respawn at their original location
    after a fixed time delay. The objective of the agent is to survive as long as possible. The agent is rewarded for
    each enemy killed.
    """

    def __init__(self,
                 env: str,
                 reward_kill: float = 1.0,
                 penalty_health_loss: float = -0.01,
                 penalty_ammo_used: float = -0.01,
                 label_enemy_reward: float = 0.001,
                 label_none_reward: float = -0.001,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.reward_kill = reward_kill
        self.penalty_health_loss = penalty_health_loss
        self.penalty_ammo_used = penalty_ammo_used
        self.label_enemy_reward = label_enemy_reward
        self.label_none_reward = label_none_reward
        self.frames_survived, self.hits_taken = 0, 0
        self.labels = np.array([0, 0, 0, 0])

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) < 2:
            return

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[1] < previous_vars[1]:
            self.hits_taken += 1
        
        # Update current labels
        self.labels = np.array([0, 0, 0, 0])

        game_state = self.game.get_state()
        if game_state is not None:
            labels = game_state.labels
            for label in labels:
                type_id = self.get_label_type_id(label)
                if type_id is not None:
                    self.labels[type_id] = 1

    def reward_wrappers_hard(self) -> List[WrapperHolder]:
        return [WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=0)]

    def reward_wrappers_easy(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=0),
            WrapperHolder(GameVariableRewardWrapper, reward=self.penalty_health_loss, var_index=1, decrease=True),
            WrapperHolder(GameVariableRewardWrapper, reward=self.penalty_ammo_used, var_index=2, decrease=True),
            # WrapperHolder(LabelRewardWrapper, enemy_reward=self.label_enemy_reward, item_reward=self.label_enemy_reward, none_reward=self.label_none_reward)
        ]

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        attack = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for t in t_left_right:
            for a in attack:
                actions.append(t + a)
        return actions

    def extra_statistics(self) -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}
        variables = self.game_variable_buffer[-1]
        return {'kills': variables[0],
                'health': variables[1],
                'ammo_left': variables[2],
                'hits_taken': self.hits_taken}

    def clear_episode_statistics(self):
        if True:
            self.evaluate_episode()

        self.frames_survived = 0
        self.hits_taken = 0
        self.labels = np.array([0, 0, 0, 0])

    def get_label_type_id(self, label):
        """
        Map an object name to a feature map.
        0 = enemy
        1 = health item
        2 = weapon
        3 = ammo
        None = anything else
        """
        ENEMY_NAME_SET = set([
            'MarineBFG', 'MarineBerserk', 'MarineChaingun', 'MarineChainsaw',
            'MarineFist', 'MarinePistol', 'MarinePlasma', 'MarineRailgun',
            'MarineRocket', 'MarineSSG', 'MarineShotgun',
            'Demon'
        ])

        HEALTH_ITEM_NAME_SET = set([
            'ArmorBonus', 'BlueArmor', 'GreenArmor', 'HealthBonus',
            'Medikit', 'Stimpack'
        ])

        WEAPON_NAME_SET = set([
            'Pistol', 'Chaingun', 'RocketLauncher', 'Shotgun', 'SuperShotgun',
            'PlasmaRifle', 'BFG9000', 'Chainsaw'
        ])

        AMMO_NAME_SET = set([
            'Cell', 'CellPack', 'Clip', 'ClipBox', 'RocketAmmo', 'RocketBox',
            'Shell', 'ShellBox'
        ])

        name = label.object_name
        value = label.value
        if value != 255 and name == 'DoomPlayer' or name in ENEMY_NAME_SET:
            return 0
        elif name in HEALTH_ITEM_NAME_SET:
            return 1
        elif name in WEAPON_NAME_SET:
            return 2
        elif name in AMMO_NAME_SET:
            return 3
        
    evaluating = True
    episode = 0
    evaluations = (0, 0)
    
    def evaluate_episode(self) -> float:
        if not self.game_variable_buffer or self.frames_survived == 0:
            return
        self.episode += 1
        print(f"--- Episode {self.episode} statistics ---")
        print("Frames survived:", self.frames_survived)
        print("Kills:", self.game_variable_buffer[-1][0])
        print("Deaths:", 1)
        print("K/D Ratio:", self.game_variable_buffer[-1][0] / 1)
        self.evaluations = (self.evaluations[0] + self.game_variable_buffer[-1][0], self.evaluations[1] + 1)
        print(f"--- Total {self.episode} statistics ---")
        print("Kills:", self.evaluations[0])
        print("Deaths:", self.evaluations[1])
        print("K/D Ratio:", self.evaluations[0] / self.evaluations[1])
        print("")
