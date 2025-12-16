import gymnasium
import numpy as np
from gymnasium.core import ObsType, WrapperObsType, RewardWrapper
from gymnasium.spaces import Box


class RescaleObservation(gymnasium.ObservationWrapper):
    """Rescale the observation space to [-1, 1]."""

    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self.observation_space = Box(low=-1, high=1, shape=self.observation_space.shape)

    def observation(self, observation: ObsType) -> WrapperObsType:
        return observation / 255. * 2 - 1


class RGBStack(gymnasium.ObservationWrapper):
    """Combine the stacked frames with RGB colours. [n_stack, h, w, 3] -> [n_stack * 3, h, w]"""

    def __init__(self, env):
        super(RGBStack, self).__init__(env)
        n_stack, height, width, channels = self.observation_space.shape
        new_shape = (n_stack * channels, height, width)
        self.observation_space = Box(
            low=np.min(self.observation_space.low),
            high=np.max(self.observation_space.high),
            shape=new_shape
        )

    def observation(self, observation: ObsType) -> WrapperObsType:
        n_stack, height, width, channels = observation.shape
        return np.transpose(observation, (0, 3, 1, 2)).reshape(n_stack * channels, height, width)


class WrapperHolder:
    """
    A wrapper holder stores a reward wrapper with its respective keyword arguments.
    """

    def __init__(self, wrapper_class, **kwargs):
        self.wrapper_class = wrapper_class
        self.kwargs = kwargs


class ConstantRewardWrapper(RewardWrapper):
    """
    Reward the agent with a constant reward at each time step.
    """

    def __init__(self, env, reward: float):
        super(ConstantRewardWrapper, self).__init__(env)
        self.rew = reward

    def reward(self, reward: float) -> float:
        return reward + self.rew


class GameVariableRewardWrapper(RewardWrapper):
    """
    Reward the agent for a change in a game variable. The agent is considered to have changed a game variable if its
    value differs from the previous frame value.
    """

    def __init__(self, env, reward: float, var_index: int = 0, decrease: bool = False):
        super(GameVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.var_index = var_index
        self.decrease = decrease

    def reward(self, reward):
        if len(self.unwrapped.game_variable_buffer) < 2:
            return reward
        vars_cur = self.unwrapped.game_variable_buffer[-1]
        vars_prev = self.unwrapped.game_variable_buffer[-2]

        var_cur = vars_cur[self.var_index]
        var_prev = vars_prev[self.var_index]

        # # Found bullets reward
        # # WrapperHolder(GameVariableRewardWrapper, reward=self.ammo_reward, var_index=2),
        # # Found shells reward
        # # WrapperHolder(GameVariableRewardWrapper, reward=self.ammo_reward, var_index=9),
        # # Found rockets reward
        # # WrapperHolder(GameVariableRewardWrapper, reward=self.ammo_reward, var_index=10),
        # # Found cells reward
        # # WrapperHolder(GameVariableRewardWrapper, reward=self.ammo_reward, var_index=11),
        # # Used bullets reward
        # # WrapperHolder(GameVariableRewardWrapper, reward=self.use_ammo_reward * 2, var_index=2, decrease=True),
        # # Used shells reward
        # # WrapperHolder(GameVariableRewardWrapper, reward=self.use_ammo_reward * 1.5, var_index=9, decrease=True),
        # # Used rockets reward
        # # WrapperHolder(GameVariableRewardWrapper, reward=self.use_ammo_reward * 2, var_index=10, decrease=True),
        # # Used cells reward
        # # WrapperHolder(GameVariableRewardWrapper, reward=self.use_ammo_reward, var_index=11, decrease=True),
        # bullet = 2
        # shell = 9
        # rocket = 10
        # cell = 11        
        # kill = 1
        # used_ammo_reward = (self.var_index == bullet or self.var_index == shell or self.var_index == rocket or self.var_index == cell) and self.decrease
        # if used_ammo_reward: 
        #     # Check if an enemy was killed. If so, do not give a penalty for using ammo.
        #     # Otherwise, the agent might learn to avoid using ammo to avoid the penalty.            
        #     killed_enemy = vars_cur[kill] > vars_prev[kill]  
        #     if killed_enemy:
        #         return reward                   

        # Apply the reward or penalty based on the change in the game variable
        if not self.decrease and var_cur > var_prev or self.decrease and var_cur < var_prev:
            reward += self.rew
        return reward


class MovementRewardWrapper(RewardWrapper):
    """
    Reward the agent for moving. Movement is measured as the distance between the agent's current location and its
    location in the previous frame.
    """

    def __init__(self, env, scaler: float):
        super(MovementRewardWrapper, self).__init__(env)
        self.scaler = scaler

    def reward(self, reward):
        if len(self.unwrapped.distance_buffer) < 2:
            return reward
        distance = self.unwrapped.distance_buffer[-1]
        reward += distance * self.scaler  # Increase the reward for movement linearly
        return reward


class AdaptiveMovementRewardWrapper(RewardWrapper):
    """
    Adaptive movement reward that decreases as the agent explores more of the map.
    Prevents endless wandering by reducing distance rewards when exploration ratio is high.
    """

    def __init__(self, env, base_scaler: float):
        super(AdaptiveMovementRewardWrapper, self).__init__(env)
        self.base_scaler = base_scaler

    def reward(self, reward):
        if len(self.unwrapped.distance_buffer) < 2:
            return reward
        
        distance = self.unwrapped.distance_buffer[-1]
        base_reward = distance * self.base_scaler
        
        # Calculate exploration ratio (visited grid cells / potential cells)
        # Assuming a typical map size, we use the number of visited cells as proxy
        exploration_ratio = 0.0
        if hasattr(self.unwrapped, 'position_grid') and len(self.unwrapped.position_grid) > 0:
            # Estimate total cells based on typical map size (e.g., 100x100 grid)
            total_cells = 10000  # 100x100 grid
            visited_cells = len(self.unwrapped.position_grid)
            exploration_ratio = min(visited_cells / total_cells, 1.0)
            self.unwrapped.exploration_ratio = exploration_ratio
        
        # Apply decay: starts at 100%, decays to minimum 20% as exploration increases
        decay_factor = max(0.2, 1.0 - min(0.8, exploration_ratio))
        
        adaptive_reward = base_reward * decay_factor
        reward += adaptive_reward
        
        return reward
