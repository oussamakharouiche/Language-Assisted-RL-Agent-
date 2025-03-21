from torch import nn, Tensor
import torch
from torch.distributions import Categorical
from .utils import layer_init


class SharedModel(nn.Module):
    """
    A neural network model where the actor (policy) and critic (value function) share layers.
    This model is useful for reinforcement learning algorithms like PPO.
    """
    def __init__(self, obs_space_shape: int, action_space_shape: int):
        """
        Initializes the shared model with common hidden layers for actor and critic.

        Args:
            obs_space_shape (int): Dimension of the observation space.
            action_space_shape (int): Dimension of the action space.
        """
        super(SharedModel, self).__init__()
        self.layer1 = layer_init(nn.Linear(obs_space_shape, 64))
        self.layer2 = layer_init(nn.Linear(64,64))
        self.actor_head = layer_init(nn.Linear(64,action_space_shape), std=0.01)
        self.critic_head = layer_init(nn.Linear(64,1), std=1.0)
        self.activation = nn.Tanh()
    
    def forward(self, obs: Tensor):
        """
        Forward pass for the model.

        Args:
            obs (torch.Tensor): The input observation tensor.

        Returns:
            Tuple[Categorical, torch.Tensor]: The action distribution and value estimation.
        """
        h = self.activation(self.layer1(obs))
        h = self.activation(self.layer2(h))

        action_distribution = Categorical(logits=self.actor_head(h))
        value = self.critic_head(h)

        return action_distribution, value

    def save(self, path: str) -> None:
        """
        Save the model state dictionary to a file.

        Args:
            path (str): The path where the state dictionary should be saved.
        """
        torch.save(self.state_dict(), path)
    

class SplitModel(nn.Module):
    """
    A neural network model where the actor and critic have separate networks.
    Useful when independent learning of policy and value function is desired.
    """
    def __init__(self, obs_space_shape: int, action_space_shape: int):
        """
        Initializes separate actor and critic networks.

        Args:
            obs_space_shape (int): Dimension of the observation space.
            action_space_shape (int): Dimension of the action space.
        """
        super(SplitModel, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_space_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,action_space_shape), std=0.01)
        )

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_space_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1), std=1.0)
        )
    
    def forward(self, obs: Tensor):
        """
        Forward pass for the split model.

        Args:
            obs (torch.Tensor): The input observation tensor.

        Returns:
            Tuple[Categorical, torch.Tensor]: The action distribution and value estimation.
        """
        pi = Categorical(logits=self.actor(obs))
        value = self.critic(obs)

        return pi, value
    
    def save(self, path: str) -> None:
        """
        Save the model state dictionary to a file.

        Args:
            path (str): The path where the state dictionary should be saved.
        """
        torch.save(self.state_dict(), path)

class MultiInputSplitModel(nn.Module):
    def __init__(self, obs_space_shapes: dict, action_space_shape: int):
        super(MultiInputSplitModel, self).__init__()

        self.bert_embedding_actor = nn.Sequential(
            layer_init(nn.Linear(obs_space_shapes["bert_embeddings"], 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(64+obs_space_shapes["grid_coordinates"], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,action_space_shape), std=0.01)
        )

        self.bert_embedding_critic = nn.Sequential(
            layer_init(nn.Linear(obs_space_shapes["bert_embeddings"], 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(64+obs_space_shapes["grid_coordinates"], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1), std=1.0)
        )

    def forward(self, obs):
        """
        Forward pass for the split model.

        Args:
            obs (torch.Tensor): The input observation tensor.

        Returns:
            Tuple[Categorical, torch.Tensor]: The action distribution and value estimation.
        """
        pi = Categorical(logits=self.actor(torch.cat([self.bert_embedding_actor(obs["bert_embeddings"]), obs["grid_coordinates"]], dim=1)))
        value = self.critic(torch.cat([self.bert_embedding_critic(obs["bert_embeddings"]), obs["grid_coordinates"]], dim=1))

        return pi, value


    def save(self, path: str) -> None:
        """
        Save the model state dictionary to a file.

        Args:
            path (str): The path where the state dictionary should be saved.
        """
        torch.save(self.state_dict(), path)



models = {
    "MlpPolicy": {
        "Shared": SharedModel,
        "Split": SplitModel,
    },
    "MultiInputPolicy": {
        "Split": MultiInputSplitModel,
    }
}