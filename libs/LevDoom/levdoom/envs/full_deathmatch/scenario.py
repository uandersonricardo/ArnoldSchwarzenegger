from collections import deque
from typing import List, Dict

import numpy as np

from levdoom.envs.base import DoomEnv
from levdoom.utils.utils import distance_traversed
from levdoom.utils.wrappers import WrapperHolder, GameVariableRewardWrapper, MovementRewardWrapper, ConstantRewardWrapper, LabelRewardWrapper


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
                 base_reward: float = 0,
                 distance_reward: float = 0.001,
                 kill_reward: float = 5.0,
                 death_reward: float = -5.0,
                 suicide_reward: float = -5.0,
                 medikit_reward: float = 0.5,
                 armor_reward: float = 0.5,
                 hit_reward: float = 1.0,
                 injured_reward: float = -0.5,
                 weapon_reward: float = 0.5,
                 ammo_reward: float = 0.1,
                 use_ammo_reward: float = -0.01,
                 label_enemy_reward: float = 0,
                 label_item_reward: float = 0,
                 label_none_reward: float = 0,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.base_reward = base_reward
        self.distance_reward = distance_reward
        self.kill_reward = kill_reward
        self.death_reward = death_reward
        self.suicide_reward = suicide_reward
        self.medikit_reward = medikit_reward
        self.armor_reward = armor_reward
        self.hit_reward = hit_reward
        self.injured_reward = injured_reward
        self.weapon_reward = weapon_reward
        self.ammo_reward = ammo_reward
        self.use_ammo_reward = use_ammo_reward
        self.label_enemy_reward = label_enemy_reward
        self.label_item_reward = label_item_reward
        self.label_none_reward = label_none_reward

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
        self.hits = 0
        self.hits_taken = 0
        self.frames_survived = 0
        self.labels = np.array([0, 0, 0, 0])

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
        if self.game.is_player_dead():
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

        # Found/used ammo
        if current_vars[2] < previous_vars[2] or current_vars[9] < previous_vars[9] or \
           current_vars[10] < previous_vars[10] or current_vars[11] < previous_vars[11]:
            self.ammo_used += 1
        elif current_vars[2] > previous_vars[2] or current_vars[9] > previous_vars[9] or \
           current_vars[10] > previous_vars[10] or current_vars[11] > previous_vars[11]:
            self.ammo_found += 1

        # Hit an enemy
        if current_vars[12] > previous_vars[12]:
            self.hits += 1
        
        # Update current labels
        self.labels = np.array([0, 0, 0, 0])

        game_state = self.game.get_state()
        if game_state is not None:
            labels = game_state.labels
            for label in labels:
                type_id = self.get_label_type_id(label)
                if type_id is not None:
                    self.labels[type_id] = 1

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        m_forward = [[0.0], [1.0]]
        attack = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward:
                for a in attack:
                    actions.append(t + m + a)
        return actions

    def reward_wrappers_easy(self) -> List[WrapperHolder]:
        return [
            # Base reward
            WrapperHolder(ConstantRewardWrapper, reward=self.base_reward),
            # Distance reward
            WrapperHolder(MovementRewardWrapper, scaler=self.distance_reward),
            # Kill reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.kill_reward, var_index=1),
            # Death reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.death_reward, var_index=8),
            # Suicide reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.suicide_reward, var_index=5, decrease=True),
            # Found medikit reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.medikit_reward, var_index=0),
            # Injured reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.injured_reward, var_index=0, decrease=True),
            # Found armor reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.armor_reward, var_index=7),
            # Found pistol reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.weapon_reward, var_index=13),
            # Found shotgun reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.weapon_reward, var_index=14),
            # Found chaingun reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.weapon_reward, var_index=15),
            # Found rocket launcher reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.weapon_reward, var_index=16),
            # Found plasma rifle reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.weapon_reward, var_index=17),
            # Found bfg9000 reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.weapon_reward, var_index=18),
            # Found bullets reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.ammo_reward, var_index=2),
            # Found shells reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.ammo_reward, var_index=9),
            # Found rockets reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.ammo_reward, var_index=10),
            # Found cells reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.ammo_reward, var_index=11),
            # Used bullets reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.use_ammo_reward, var_index=2, decrease=True),
            # Used shells reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.use_ammo_reward, var_index=9, decrease=True),
            # Used rockets reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.use_ammo_reward, var_index=10, decrease=True),
            # Used cells reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.use_ammo_reward, var_index=11, decrease=True),
            # Hit enemy reward
            WrapperHolder(GameVariableRewardWrapper, reward=self.hit_reward, var_index=12),
            # Enemy on screen reward
            WrapperHolder(LabelRewardWrapper, enemy_reward=self.label_enemy_reward, item_reward=self.label_item_reward, none_reward=self.label_none_reward)
        ]

    def reward_wrappers_hard(self) -> List[WrapperHolder]:
        return [WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=0)]

    def extra_statistics(self) -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}

        variables = self.game_variable_buffer[-1]

        return {
            'health': variables[0],
            'kills': self.kills,
            'deaths': self.deaths,
            'suicides': self.suicides,
            'frags': self.frags,
            'medikits': self.medikits,
            'armors': self.armors,
            'pistol': self.pistol,
            'shotgun': self.shotgun,
            'chaingun': self.chaingun,
            'rocket_launcher': self.rocket_launcher,
            'plasma_rifle': self.plasma_rifle,
            'bfg9000': self.bfg9000,
            'bullets': self.bullets,
            'shells': self.shells,
            'rockets': self.rockets,
            'cells': self.cells,
            'frames_survived': self.frames_survived,
            'movement': np.mean(self.distance_buffer).round(3),
            'hits': self.hits,
            'hits_taken': self.hits_taken,
            'ammo_used': self.ammo_used,
            'ammo_found': self.ammo_found,
            'labels': self.labels
        }

    def clear_episode_statistics(self):
        if True:
            self.evaluate_episode()

        super().clear_episode_statistics()

        self.distance_buffer.clear()
        self.ammo_used = 0
        self.ammo_found = 0
        self.hits = 0
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
        self.labels = np.array([0, 0, 0, 0])
    
    def print_state(self, vars: List[float]) -> None:
        print(f' Health: {vars[0]} | Kills: {vars[1]} | Ammo: {vars[2]} | Position: ({vars[3]:.2f}, {vars[4]:.2f}) | Frags: {vars[5]} | Selected Weapon: {vars[6]} | Armor: {vars[7]} | Dead: {vars[8]} | Bullets: {vars[2]} | Shells: {vars[9]} | Rockets: {vars[10]} | Cells: {vars[11]}')

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
    evaluations = (0, 0, 0, 0, 0)
    
    def evaluate_episode(self) -> float:
        if not self.game_variable_buffer or self.frames_survived == 0:
            return
        self.episode += 1
        print(f"--- Episode {self.episode} statistics ---")
        print("Frames survived:", self.frames_survived)
        print("Kills:", self.kills)
        print("Deaths:", 1)
        print("Suicides:", self.suicides)
        print("Items:", self.medikits + self.armors + self.ammo_found)
        print("Weapons found:", self.pistol + self.shotgun + self.chaingun + self.rocket_launcher + self.plasma_rifle + self.bfg9000)
        print("K/D Ratio:", self.kills / 1)
        self.evaluations = (self.evaluations[0] + self.kills, self.evaluations[1] + 1, self.evaluations[2] + self.suicides,
                            self.evaluations[3] + self.medikits + self.armors + self.ammo_found,
                            self.evaluations[4] + self.pistol + self.shotgun + self.chaingun + self.rocket_launcher + self.plasma_rifle + self.bfg9000)
        print(f"--- Total {self.episode} statistics ---")
        print("Kills:", self.evaluations[0])
        print("Deaths:", self.evaluations[1])
        print("K/D Ratio:", self.evaluations[0] / self.evaluations[1])
        print("Suicides:", self.evaluations[2])
        print("Items:", self.evaluations[3])
        print("Weapons found:", self.evaluations[4])
        print("")
