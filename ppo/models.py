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
            layer_init(nn.Linear(*obs_space_shapes["bert_embeddings"], 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(64+(obs_space_shapes["grid_coordinates"][0]), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,action_space_shape), std=0.01)
        )

        self.bert_embedding_critic = nn.Sequential(
            layer_init(nn.Linear(*obs_space_shapes["bert_embeddings"], 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(64+(obs_space_shapes["grid_coordinates"][0]), 64)),
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


class MultiInputSharedModelAttention(nn.Module):
    def __init__(self, obs_space_shapes: dict, action_space_shape: int):
        super(MultiInputSharedModelAttention, self).__init__()

        self.text_to_key = nn.Sequential(
            layer_init(nn.Linear(*obs_space_shapes["bert_embeddings"], 64))
        )

        self.text_to_value = nn.Sequential(
            layer_init(nn.Linear(*obs_space_shapes["bert_embeddings"], 32))
        )

        self.cnn_query = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, 3, padding=1)
        )

        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(4, 16, 3, padding=1),
            nn.Tanh()
        )

        self.actor = nn.Sequential(
            nn.Conv2d(16, 4, 1),
            nn.Flatten(),
            nn.Linear(4*10*10, 64), 
            nn.Tanh(),
            nn.Linear(64, action_space_shape)
        )

        self.critic = nn.Sequential(
            nn.Conv2d(16, 4, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(4*10*10, 64), 
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        """
        Forward pass for the split model.

        Args:
            obs (torch.Tensor): The input observation tensor.

        Returns:
            Tuple[Categorical, torch.Tensor]: The action distribution and value estimation.
        """
        B = obs["grid"].size(0)
        H = obs["grid"].size(2)

        queries = self.cnn_query(obs["grid"])
        queries = queries.flatten(2).permute(0, 2, 1)

        keys = self.text_to_key(obs["bert_embeddings"])
        
        attn_scores = torch.einsum('bqd,bd->bq', queries, keys)
        
        attn_weights = torch.softmax(attn_scores / (64 ** 0.5), dim=-1)

        attended = attn_weights.view(B, H, H).unsqueeze(1)

        features = self.shared_conv(attended)

        pi = Categorical(logits=self.actor(features))
        value = self.critic(features)

        return pi, value


    def save(self, path: str) -> None:
        """
        Save the model state dictionary to a file.

        Args:
            path (str): The path where the state dictionary should be saved.
        """
        torch.save(self.state_dict(), path)


class MultiInputSharedModelMultiHeadAttention(nn.Module):
    def __init__(self, obs_space_shapes: dict, action_space_shape: int):
        super(MultiInputSharedModelMultiHeadAttention, self).__init__()

        self.num_heads = 4
        self.head_dim = 64 // self.num_heads

        self.text_to_key = nn.Sequential(
            layer_init(nn.Linear(*obs_space_shapes["bert_embeddings"], 64))
        )

        self.cnn_query = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, 3, padding=1)
        )

        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.num_heads, 32, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Tanh()
        )

        self.actor = nn.Sequential(
            nn.Conv2d(16, 4, 1),
            nn.Flatten(),
            nn.Linear(4*10*10, 64), 
            nn.Tanh(),
            nn.Linear(64, action_space_shape)
        )

        self.critic = nn.Sequential(
            nn.Conv2d(16, 4, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(4*10*10, 64), 
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        """
        Forward pass for the split model.

        Args:
            obs (torch.Tensor): The input observation tensor.

        Returns:
            Tuple[Categorical, torch.Tensor]: The action distribution and value estimation.
        """
        B = obs["grid"].shape[0]
        H = obs["grid"].size(2)

        queries = self.cnn_query(obs["grid"])
        queries = queries.flatten(2).view(B, self.num_heads, self.head_dim, H*H).permute(0,1,3,2)

        keys = self.text_to_key(obs["bert_embeddings"]).view(B, self.num_heads, self.head_dim)

        attn_scores = torch.einsum('bhqd,bhd->bhq', queries, keys)
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        
        attended_spatial = attn_weights.view(B, self.num_heads, H, H)

        features = self.shared_conv(attended_spatial)

        pi = Categorical(logits=self.actor(features))
        value = self.critic(features)

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
    },
    "MultiInputAttentionPolicy": {
        "Shared": MultiInputSharedModelAttention,
    },
    "MultiInputMultiHeadAttentionPolicy": {
        "Shared": MultiInputSharedModelMultiHeadAttention,
    }, 
}