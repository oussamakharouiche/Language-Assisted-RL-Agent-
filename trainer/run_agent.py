import gymnasium as gym
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.env_with_language import GridWorldEnv  
def run_agent(model_path, grid_size=10, num_episodes=10, render_mode='human'):
    """
    Loads a trained PPO model and evaluates it in the GridWorld environment.

    Args:
        model_path: The path to the saved model.
        grid_size: The size of the grid for the environment.
        num_episodes: The number of episodes to run for evaluation.
        render_mode: The mode to render the environment ('human', 'rgb_array', or 'ansi').
    """
    # Create the GridWorld environment
    env = GridWorldEnv(grid_size=grid_size, render_mode='human')

    # Load the trained model
    model = PPO.load(model_path, env=env)


    # Optional: Demonstrate the trained agent
    obs, info = env.reset()  # Reset with a prompt
    for _ in range(1000):  # Run for a few steps
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()

    

def main():
    # Specify the path to the saved model
    model_path = "../models/text_guided_2"  # Update this path as needed
    run_agent(model_path)

if __name__ == "__main__":
    main()
