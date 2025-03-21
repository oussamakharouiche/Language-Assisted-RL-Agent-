import gymnasium as gym
import argparse
import sys
import os
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment import GridWorldEnv, LanguageGridWorldEnv
from embeddings import *
from ppo.ppo import Trainer
from ppo.utils import load_config


def make_env():
    return GridWorldEnv()

def make_language_env(data_path="./dataset/data.pickle"):
    return LanguageGridWorldEnv(embed_text, EMBED_TEXT_DIM, data_path=data_path)

def train_agent(make_env, params):
    """
    Trains a PPO agent on the GridWorld environment.

    Args:
        env: The GridWorld environment instance.
        total_timesteps: The total number of training timesteps.
        grid_size: Size of gridworld for model saving.

    Returns:
        The trained PPO model.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    exp_name = params["exp_name"]
    writer = SummaryWriter(f"logs/{exp_name}")

    trainer = Trainer(writer, params, make_env)

    trainer.training_loop()

    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_path = os.path.join(models_dir, params["model_name"])
    trainer.model.save(model_path)
    print(f"Model saved to {model_path}")

    print(trainer.test_policy(make_env))

    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="The path to the YAML config file for hyperparameters")
    args = parser.parse_args()

    params = load_config(args.config_path)

    train_agent(make_env if params["type"] == "simple" else make_language_env, params)


if __name__ == "__main__":
    main()