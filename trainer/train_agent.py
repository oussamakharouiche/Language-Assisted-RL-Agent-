import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.env_with_language import GridWorldEnv   


def make_env():
    return GridWorldEnv()

def train_agent(env, total_timesteps=100000, grid_size=10):
    """
    Trains a PPO agent on the GridWorld environment.

    Args:
        env: The GridWorld environment instance.
        total_timesteps: The total number of training timesteps.
        grid_size: Size of gridworld for model saving.

    Returns:
        The trained PPO model.
    """

    vec_env = make_vec_env(make_env, n_envs=1)

    # Specify the logging directory for TensorBoard
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Instantiate the agent with the tensorboard log directory
    # model = PPO("MlpPolicy", vec_env, tensorboard_log=log_dir,verbose=2)
    model = PPO("MultiInputPolicy", vec_env, tensorboard_log=log_dir,verbose=2)
    

    # Train the agent and write logs
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"PPO_GridWorld_{grid_size}x{grid_size}")

    # Create the directory for saving models if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Save the trained model
    model_path = os.path.join(models_dir, f"PPO_GridWorld_{grid_size}x{grid_size}_{total_timesteps}")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model


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
    render_mode = "ansi" # "human" or "rgb_array" or "ansi"
    env = GridWorldEnv(grid_size=grid_size, render_mode=render_mode)
    

    #Train the model
    trained_model = train_agent(env,total_timesteps,grid_size)

    # Evaluate trained model
    env = GridWorldEnv(grid_size=grid_size, render_mode=render_mode)
    evaluate_trained_agent(trained_model,env)

    # Reset and get observation.
    obs, info = env.reset()  # Use the new reset signature
    for _ in range(1000):  # Run for a few steps
        action, _states = trained_model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()