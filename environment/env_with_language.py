import numpy as np
import pandas as pd
import gymnasium as gym
import pygame
import random
import torch
from transformers import BertTokenizer, BertModel

class GridWorldEnv(gym.Env):
    """
    A Grid World Environment with BERT-based goal embeddings.

    The environment simulates an agent navigating a grid. The goal cell is associated with a text
    prompt, whose embedding is computed using a pre-trained BERT model. The agent can move in four
    cardinal directions or perform a no-op action to check for goal achievement.

    Action Space:
        0: Up
        1: Down
        2: Left
        3: Right
        4: No-op (check for goal)

    Observation Space:
        {
            "grid_coordinates": Agent's (row, column) position.
            "bert_embeddings": BERT embedding of the goal text.
        }

    Rendering Modes:
        - human: Uses pygame to render the grid.
        - rgb_array: Returns the rendered frame as an RGB numpy array.
        - ansi: Provides a text-based grid representation.
    """
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_size=10, render_mode="human", data_path="../dataset/data.pickle"):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.nb_steps = 0 
        self.data = pd.read_pickle(data_path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        # Define the action space : # 0 = Up, 1 = Down, 2 = Left, 3 = Right, 4 = No-op
        self.action_space = gym.spaces.Discrete(5)

        # Define the observation space
        self.observation_space = gym.spaces.Dict({
            "grid_coordinates": gym.spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
            "bert_embeddings": gym.spaces.Box(low=-1.0, high=1.0, shape=(768,), dtype=np.float32),
        })

        self.start_pos = None
        self.goal_pos = None
        self.agent_pos = None

        # Pygame setup

        self.window_size = 512  # Pygame window size
        self._agent_color = (0, 0, 255)  # Blue
        self._goal_color = (0, 255, 0)   # Green
        self._start_color = (255, 0, 0)  # Red
        self._background_color = (255, 255, 255) #white
        self._grid_line_color = (0,0,0) #black
        self.window = None
        self.clock = None
        self._cell_size = self.window_size // self.grid_size


    def _get_obs(self):
        return {
        "grid_coordinates": self.agent_pos,
        "bert_embeddings": np.array(self.goal_emb),
    }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.nb_steps = 0
        self.start_pos = self._random_pos()
        self.goal_pos = random.choice([(self.grid_size-1, self.grid_size-1), (0, self.grid_size-1), (self.grid_size-1, 0)])
        self.text_goal = random.choice(list(self.data[(self.data["row"] == self.goal_pos[0]) & (self.data["column"] == self.goal_pos[1])]["prompt"]))
        with torch.no_grad():
            self.goal_emb = self.model(**self.tokenizer(self.text_goal, return_tensors="pt", padding=True, truncation=True)).last_hidden_state
            # self.goal_emb = torch.mean(self.goal_emb, dim=1)
            self.goal_emb = self.goal_emb[:, 0, :] ## cls token 

        self.agent_pos = self.start_pos

        if self.render_mode in ["human", "rgb_array"]:
            self._render_frame()  # Initial render
        
        return self._get_obs(), {"start_pos": self.start_pos, "goal_pos": self.goal_pos}

    def _random_pos(self):
        return (self.np_random.integers(0, self.grid_size), self.np_random.integers(0, self.grid_size))

    def step(self, action):
        self._agent_color = (0, 0, 255)
        self.nb_steps += 1
        terminated = False
        reward = -1
        row, col = self.agent_pos
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.grid_size - 1, col + 1)
        elif action == 4:
            self._agent_color = (255,255,0) #Yellow
            if self.agent_pos == self.goal_pos :
                reward = 60 
                terminated = True 

        else:
            raise ValueError("Invalid action")

        self.agent_pos = (row, col)
        truncated = (self.nb_steps == 50)
        info = {}

        if self.render_mode in ["human", "rgb_array"]:
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "ansi":
             print(self._render_text())
    
    def _render_text(self):
        grid = np.full((self.grid_size, self.grid_size), ".")
        grid[self.start_pos] = "S"
        grid[self.goal_pos] = "G"
        grid[self.agent_pos] = "A"
        return "\n".join(["|".join(row) for row in grid])

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self._background_color)

        # Draw goal
        pygame.draw.rect(
            canvas,
            self._goal_color,
            pygame.Rect(
                (self.goal_pos[1] * self._cell_size, self.goal_pos[0] * self._cell_size),
                (self._cell_size, self._cell_size),
            ),
        )
        
        # Draw start position
        pygame.draw.rect(
            canvas,
            self._start_color,  # Red for start
            pygame.Rect(
                (self.start_pos[1] * self._cell_size, self.start_pos[0] * self._cell_size),
                (self._cell_size, self._cell_size),
            ),
        )

        # Draw agent
        agent_radius = self._cell_size // 3  # Make agent a circle
        pygame.draw.circle(
            canvas,
            self._agent_color,
            (self.agent_pos[1] * self._cell_size + self._cell_size // 2,
             self.agent_pos[0] * self._cell_size + self._cell_size // 2),
            agent_radius,
        )

        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                self._grid_line_color,
                (0, i * self._cell_size),
                (self.window_size, i * self._cell_size),
                width=2  # Thicker lines
            )
            pygame.draw.line(
                canvas,
                self._grid_line_color,
                (i * self._cell_size, 0),
                (i * self._cell_size, self.window_size),
                width=2
            )

        if self.render_mode == "human":
             # Update the display
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump() # Process events
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

