import numpy as np
import gymnasium as gym
import pygame
import random


class GridWorldEnv(gym.Env):
    """
    A Grid World Environment for Reinforcement Learning tasks using Gymnasium.

    The environment is represented as a grid where an agent starts at a random position
    and must navigate toward a goal located at one of the four corners. The agent can move
    in four directions (up, down, left, right) or perform a no-operation (no-op) action.
    Pygame is used for rendering the environment.

    Attributes:
        grid_size (int): The number of rows and columns in the grid.
        render_mode (str): The mode of rendering ('human', 'ansi', or 'rgb_array').
        nb_steps (int): Counter for the number of steps taken during the episode.
        action_space (gym.spaces.Discrete): The space of possible actions.
        observation_space (gym.spaces.Box): The space of observations (agent and goal positions).
        start_pos (tuple): The starting position of the agent.
        goal_pos (tuple): The goal position.
        agent_pos (tuple): The current position of the agent.
        window_size (int): The pixel size of the Pygame window.
        _agent_color (tuple): RGB color for the agent.
        _goal_color (tuple): RGB color for the goal.
        _start_color (tuple): RGB color for the start position.
        _background_color (tuple): RGB color for the window background.
        _grid_line_color (tuple): RGB color for the grid lines.
        window: Pygame display surface.
        clock: Pygame clock for controlling the frame rate.
        _cell_size (int): Pixel size of each grid cell (derived from window_size and grid_size).
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_size=10, render_mode="human"):
        """
        Initialize the GridWorld Environment.

        Args:
            grid_size (int): Size of the grid (default is 10).
            render_mode (str): The rendering mode ("human", "ansi", or "rgb_array").
        """
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.nb_steps = 0  # Reset step counter for each episode

        # Define the action space:
        # 0 = Up, 1 = Down, 2 = Left, 3 = Right, 4 = No-op
        self.action_space = gym.spaces.Discrete(5)

        # Define the observation space as a concatenation of agent and goal positions.
        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32
        )

        self.start_pos = None
        self.goal_pos = None
        self.agent_pos = None

        # Pygame window setup parameters
        self.window_size = 512  # Pixel dimensions of the Pygame window
        self._agent_color = (0, 0, 255)  # Blue color for the agent
        self._goal_color = (0, 255, 0)  # Green color for the goal
        self._start_color = (255, 0, 0)  # Red color for the start position
        self._background_color = (255, 255, 255)  # White background color
        self._grid_line_color = (0, 0, 0)  # Black color for the grid lines

        self.window = None  # Pygame window surface
        self.clock = None  # Pygame clock to control frame rate
        self._cell_size = (
            self.window_size // self.grid_size
        )  # Compute cell size in pixels

    def _get_obs(self):
        """
        Construct and return the current observation.

        Combines the agent's position and the goal's position into a single array.

        Returns:
            numpy.ndarray: A 1D array of length 4 containing [agent_row, agent_col, goal_row, goal_col].
        """
        return np.concatenate(
            (np.array(self.agent_pos), np.array(self.goal_pos)), axis=None
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.

        The starting position is randomly generated and the goal is randomly chosen
        from one of the four corners of the grid.

        Args:
            seed (int, optional): Seed for random number generation.

        Returns:
            tuple: A tuple (observation, info) where observation is the initial state and
                   a dictionary containing the start and goal positions.
        """
        super().reset(seed=seed)
        self.nb_steps = 0
        self.start_pos = self._random_pos()
        self.goal_pos = random.choice(
            [
                (0, 0),
                (self.grid_size - 1, self.grid_size - 1),
                (0, self.grid_size - 1),
                (self.grid_size - 1, 0),
            ]
        )
        self.agent_pos = self.start_pos

        if self.render_mode in ["human", "rgb_array"]:
            self._render_frame()  # Render the initial frame if necessary
        return self._get_obs(), {"start_pos": self.start_pos, "goal_pos": self.goal_pos}

    def _random_pos(self):
        """
        Generate a random position within the grid boundaries.

        Uses the environment's random number generator.

        Returns:
            tuple: A tuple (row, col) representing a random position in the grid.
        """
        return (
            self.np_random.integers(0, self.grid_size),
            self.np_random.integers(0, self.grid_size),
        )

    def step(self, action):
<<<<<<< HEAD
        """
        Execute one time step within the environment based on the given action.

        Updates the agent's position according to the action, computes the reward, and
        determines whether the episode has terminated or been truncated.

        Args:
            action (int): An integer representing the action to take.
                          0 = Up, 1 = Down, 2 = Left, 3 = Right, 4 = No-op.

        Returns:
            tuple: A tuple containing:
                - observation (numpy.ndarray): The new state of the environment.
                - reward (int): The reward received after taking the action.
                - terminated (bool): Flag indicating whether the episode has ended.
                - truncated (bool): Flag indicating whether the episode has been truncated (e.g., due to a step limit).
                - info (dict): Additional information (empty in this implementation).
        """
        # Reset agent color to blue for movement actions
=======
>>>>>>> d93ec9c6324ab82bcd4415802cccd1ae8394ce2e
        self._agent_color = (0, 0, 255)
        self.nb_steps += 1
        terminated = False
        reward = -1  # Default reward for a move

        row, col = self.agent_pos
        if action == 0:  # Move up
            row = max(0, row - 1)
        elif action == 1:  # Move down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # Move left
            col = max(0, col - 1)
        elif action == 3:  # Move right
            col = min(self.grid_size - 1, col + 1)
<<<<<<< HEAD
        elif action == 4:  # No-op action
            self._agent_color = (255, 255, 0)  # Change color to yellow for no-op
            if self.agent_pos == self.goal_pos:
                reward = 60  # Reward for reaching the goal and performing no-op
                terminated = True
=======
        elif action == 4:
            self._agent_color = (255,255,0) #Yellow
            if self.agent_pos == self.goal_pos :
                reward = 60 
                terminated = True 
>>>>>>> d93ec9c6324ab82bcd4415802cccd1ae8394ce2e

        else:
            raise ValueError("Invalid action")

        self.agent_pos = (row, col)
        truncated = self.nb_steps == 50  # Episode truncation after 50 steps
        info = {}

        if self.render_mode in ["human", "rgb_array"]:
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """
        Render the current state of the environment.

        Depending on the render mode, the environment is rendered as an RGB array,
        to a Pygame window, or as a text-based representation.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "ansi":
            print(self._render_text())

    def _render_text(self):
        """
        Generate a text-based representation of the grid.

        The grid is represented using characters: 'S' for the start, 'G' for the goal,
        'A' for the agent, and '.' for empty cells.

        Returns:
            str: A string representation of the grid.
        """
        grid = np.full((self.grid_size, self.grid_size), ".")
        grid[self.start_pos] = "S"
        grid[self.goal_pos] = "G"
        grid[self.agent_pos] = "A"
        return "\n".join(["|".join(row) for row in grid])

    def _render_frame(self):
        """
        Render the environment's current state using Pygame.

        Draws the background, grid lines, goal, start position, and agent onto a canvas,
        then updates the display based on the render mode.

        Returns:
            numpy.ndarray or None: If the render mode is "rgb_array", returns the RGB array of
                                     the frame; otherwise, returns None.
        """
        # Initialize the Pygame window if not already created
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create a canvas and fill it with the background color
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self._background_color)

        # Draw the goal as a rectangle on the canvas
        pygame.draw.rect(
            canvas,
            self._goal_color,
            pygame.Rect(
                (
                    self.goal_pos[1] * self._cell_size,
                    self.goal_pos[0] * self._cell_size,
                ),
                (self._cell_size, self._cell_size),
            ),
        )

        # Draw the start position as a rectangle
        pygame.draw.rect(
            canvas,
            self._start_color,
            pygame.Rect(
                (
                    self.start_pos[1] * self._cell_size,
                    self.start_pos[0] * self._cell_size,
                ),
                (self._cell_size, self._cell_size),
            ),
        )

        # Draw the agent as a circle
        agent_radius = self._cell_size // 3  # Make agent a circle
        pygame.draw.circle(
            canvas,
            self._agent_color,
            (
                self.agent_pos[1] * self._cell_size + self._cell_size // 2,
                self.agent_pos[0] * self._cell_size + self._cell_size // 2,
            ),
            agent_radius,
        )

        # Draw grid lines to delineate cells
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                self._grid_line_color,
                (0, i * self._cell_size),
                (self.window_size, i * self._cell_size),
                width=2,  # Thicker lines
            )
            pygame.draw.line(
                canvas,
                self._grid_line_color,
                (i * self._cell_size, 0),
                (i * self._cell_size, self.window_size),
                width=2,
            )

        if self.render_mode == "human":
            # Update the display for human viewing
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()  # Process events
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # For rgb_array render mode
            # Return the current frame as an RGB array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Close the environment and clean up resources.

        Shuts down the Pygame display and quits Pygame to free up system resources.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
