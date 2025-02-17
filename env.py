import numpy as np
import gymnasium as gym
import random

# Gridworld environment
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    def __init__(self, grid_size=10 , render_mode = "human"):
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Define the action space : # 0 = Up, 1 = Down, 2 = Left, 3 = Right
        self.action_space = gym.spaces.Discrete(4)
        
        # Define the observation space
        self.observation_space = gym.spaces.Box(low = 0, high = self.grid_size-1,
                                                      shape = (2,), dtype = np.int32)
        
        self.start_pos = None 
        self.goal_pos = None 
        self.agent_pos = None

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.int32)
    
    def reset(self):
        self.start_pos = self._random_pos()
        self.goal_pos = self._random_pos()
        while self.start_pos == self.goal_pos:
            self.goal_pos = self._random_pos()
        self.agent_pos = self.start_pos

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"start_pos": self.start_pos, "goal_pos": self.goal_pos}
    
    def _random_pos(self):
        return (self.np_random.integers(0, self.grid_size), self.np_random.integers(0, self.grid_size))
    
    def step(self, action):
        row, col = self.agent_pos
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.grid_size - 1, col + 1)
        else:
            raise ValueError("Invalid action")

        self.agent_pos = (row, col)
        truncated = False
        reward = 0
        info = {}
        if self.agent_pos == self.goal_pos:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode = "human"):
        if mode == "human":
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (i, j) == self.agent_pos:
                        print("A", end="")
                    elif (i, j) == self.start_pos:
                        print("S", end="")
                    elif (i, j) == self.goal_pos:
                        print("G", end="")
                    else:
                        print(".", end="")
                print()
            print()
        else:
            raise ValueError("Invalid render mode")
        
