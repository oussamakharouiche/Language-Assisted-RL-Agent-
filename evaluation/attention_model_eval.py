import gymnasium as gym
import argparse
import sys
import os
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment import GridWorldEnv, LanguageGridWorldEnv, LanguageImgGridWorldEnv
from embeddings import *
from ppo.ppo import Trainer
from ppo.utils import load_config
from ppo.models import MultiInputSharedModelMultiHeadAttention
import torch

def make_language_seq_env_attention(data_path="./dataset/data.pickle"):
    return LanguageImgGridWorldEnv(embed_text_seq_classification, EMBED_TEXT_SEQ_CLASSIFICATION_DIM, data_path=data_path)

def get_attended_spatial(model, obs):
    """
    Captures attended_spatial (the input to shared_conv) using a forward hook.
    """
    attended_spatial_value = None  # Variable to store attended_spatial

    def hook_fn(module, input, output):
        nonlocal attended_spatial_value
        attended_spatial_value = input[0]  # The first input to shared_conv

    # Register the hook on `shared_conv`
    hook = model.shared_conv.register_forward_hook(hook_fn)

    # Run forward pass
    with torch.no_grad():
        _ = model(obs)

    # Remove the hook after capturing the value
    hook.remove()

    return attended_spatial_value


def eval_agent(make_env, params):
    env = make_env(data_path="./dataset/data_test.pickle")
    
    model = MultiInputSharedModelMultiHeadAttention({"bert_embeddings": (EMBED_TEXT_SEQ_CLASSIFICATION_DIM,)}, 5)

    model.load_state_dict(torch.load("./models/multi_head_attention"))
    
    state, _ = env.reset()

    state = {
        k: torch.tensor(v, dtype=torch.float).unsqueeze(0) for k,v in state.items()
    }
    return env.text_goal, env.agent_pos ,get_attended_spatial(model, state)

if __name__ == "__main__":
    goal,pos, attn = eval_agent(make_language_seq_env_attention, load_config("./configs/Language_seq_multi_head_attention_env.yaml"))
    import pdb;pdb.set_trace()