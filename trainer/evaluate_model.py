import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.env_with_language import GridWorldEnv   


def evaluate_trained_agent(model, env, num_episodes=10):
    """
    Evaluates the trained agent.

    Args:
        model: The trained PPO model.
        env:  GridWorld environment.
        num_episodes: The number of episodes to run for evaluation.
    """

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")



def main():
    grid_size = 10  # Set the grid size for your environment
    total_timesteps = 1000000
    render_mode = "human" # "human" or "rgb_array" or "ansi"
    env = GridWorldEnv(grid_size=grid_size, render_mode=render_mode, data_path="./dataset/data_test.pickle")

    trained_model = PPO.load("./text_guided.zip", env=env)

    # Evaluate trained model
    env = GridWorldEnv(grid_size=grid_size, render_mode=render_mode)
    evaluate_trained_agent(trained_model,env, num_episodes=1200)


    # #---- Optional:  Demonstrate the trained agent (requires human render mode)-----#
    # # Load model
    # # models_dir = "models"
    # # model_path = os.path.join(models_dir, f"PPO_GridWorld_{grid_size}x{grid_size}_{total_timesteps}")
    # # trained_model = PPO.load(model_path, env=env)

    # # Reset and get observation.
    # obs, info = env.reset()  # Use the new reset signature
    # for _ in range(1000):  # Run for a few steps
    #     action, _states = trained_model.predict(obs, deterministic=True)
    #     obs, rewards, terminated, truncated, info = env.step(action)
    #     env.render()
    #     if terminated or truncated:
    #         obs, info = env.reset()
    # env.close()


if __name__ == "__main__":
    main()