from typing import Dict
import numpy as np

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, state_dim: tuple, size: int, batch_size: int = 32):
        self.batch_size = batch_size
        self.buffer_capacity = size
        self.current_size = 0
        self.count = 0
        self.buffer = {
            'state': np.zeros((self.buffer_capacity,) + state_dim, dtype=np.uint8),
            'next_state': np.zeros((self.buffer_capacity,) + state_dim, dtype=np.uint8),
            'action': np.zeros(self.buffer_capacity, dtype=np.int32),
            'reward': np.zeros(self.buffer_capacity, dtype=np.float32),
            'done': np.zeros(self.buffer_capacity, dtype=np.bool),
        }

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer['state'][self.count] = state
        self.buffer['next_state'][self.count] = next_state
        self.buffer['action'][self.count] = action
        self.buffer['reward'][self.count] = reward
        self.buffer['done'][self.count] = done
        self.count = (self.count + 1) % self.buffer_capacity
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        return Dict(state=self.buffer['state'][idxs],
                    next_state=self.buffer['next_state'][idxs],
                    action=self.buffer['action'][idxs],
                    reward=self.buffer['reward'][idxs],
                    done=self.buffer['done'][idxs])

    def __len__(self) -> int:
        return self.current_size
