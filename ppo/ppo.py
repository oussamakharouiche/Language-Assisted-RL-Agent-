import time
import numpy as np
from tqdm import tqdm
import os
import torch
from stable_baselines3.common.env_util import make_vec_env
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from .models import models
from .utils import normalize
    
class Trainer:
    """
    Trainer class for running PPO training on a Gym environment.
    """

    def __init__(
        self, 
        writer: SummaryWriter,
        params: dict,
        make_env
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            config (dict): Dictionary containing training hyperparameters.
            writer (SummaryWriter): TensorBoard writer for logging.
        """
        self.seed = params["seed"]
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.exception =  ["MultiInputPolicy", "MultiInputAttentionPolicy", "MultiInputMultiHeadAttentionPolicy"]

        self.num_steps = params["num_steps"]
        self.policy =  params["policy"]
        self.n_envs = params["n_envs"]
        self.env = make_vec_env(make_env, n_envs=self.n_envs, seed=self.seed)
        self.obs_space_shape = {key:self.env.observation_space[key].shape for key in self.env.observation_space} if self.policy in self.exception else self.env.observation_space.shape
        self.model = models[self.policy]["Shared" if params["share"] else "Split"](self.obs_space_shape, self.env.action_space.n)
        # self.model = SharedModel(self.env.observation_space.shape[0], self.env.action_space.n) if params["share"] else SplitModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.batch_size = self.num_steps*self.n_envs
        self.mini_batch_size = self.batch_size//params["num_minibatches"]
        self.gamma = params["gamma"]
        self.gae_lambda = params["gae_lambda"]
        self.clip_eps = params["clip_eps"]
        self.c1 = params["vf_coeff"]
        self.c2 = params["ent_coeff"]
        self.n_epochs = params["n_epochs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() and params["cuda"] else "cpu")
        self.model.to(self.device)
        if self.policy in self.exception:
            self.obs = self.to_tensor_dict(self.env.reset())
        else:
            self.obs = self.to_tensor(self.env.reset()).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["learning_rate"], eps=1e-5)
        self.update = (params["total_timestamp"]+self.batch_size-1)//self.batch_size
        self.writer = writer
        self.loss_step = 0
        self.reward_step = 0
        self.global_step = 0
        self.anneal_lr = params["anneal_lr"]
        self.lr = params["learning_rate"]
        self.model_name = params["model_name"]

    def to_tensor(self,arr: np.ndarray) -> torch.Tensor:
        """
        Convert an array-like object to a PyTorch tensor.

        Args:
            arr: Input array.

        Returns:
            torch.Tensor: Converted tensor.
        """
        return torch.tensor(arr, dtype=torch.float)
    
    def to_tensor_dict(self, data: dict, test: bool = False) -> dict:
        """
        Convert a dictionary of array-like objects to a dictionary of PyTorch tensors.
        This function minimizes the overhead by doing the conversion in one go.
        
        Args:
            data (dict): Dictionary of observations.
        
        Returns:
            dict: Dictionary of converted tensors on the proper device.
        """
        if test:
            return {key: torch.tensor(value, dtype=torch.float, device=self.device).unsqueeze(0)
                for key, value in data.items()}
        return {key: torch.tensor(value, dtype=torch.float, device=self.device)
                for key, value in data.items()}
        
    @torch.no_grad()
    def sample(self) -> dict:
        """
        Collect samples from the environment using the current policy.

        Returns:
            dict: Dictionary containing observations, actions, values, log probabilities,
                  advantages, and rewards.
        """
        rewards = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)
        actions = torch.zeros((self.n_envs, self.num_steps), dtype=torch.long)
        dones = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)
        observations = {key:torch.zeros((self.n_envs, self.num_steps, *value), dtype=torch.float) for key, value in self.obs_space_shape.items()} if self.policy in self.exception else torch.zeros((self.n_envs, self.num_steps, *self.obs_space_shape), dtype=torch.float)
        values = torch.zeros((self.n_envs, self.num_steps+1), dtype=torch.float)
        log_probs = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)

        env_interaction_time = 0.0
        convert_time = 0.0
        for t in range(self.num_steps):
            if self.policy in self.exception:
                for key in self.obs:
                    observations[key][:,t] = self.obs[key]
            else:
                observations[:,t] = self.obs
            
            action_distribution,v = self.model(self.obs)
            action = action_distribution.sample() ### sample epsilon greedy
            actions[:,t] = action
            values[:,t] = v.reshape(self.n_envs,).detach()
            log_probs[:,t] = action_distribution.log_prob(action).detach() ### check the format

            start_step = time.time()
            self.obs, reward, done, info =  self.env.step(action.cpu().numpy())
            env_interaction_time += time.time() - start_step
            start_convert = time.time()
            if self.policy in self.exception:
                self.obs = self.to_tensor_dict(self.obs)
            else:
                self.obs = self.to_tensor(self.obs).to(self.device)
            convert_time += time.time() - start_convert
            dones[:,t] = self.to_tensor(done)
            rewards[:,t] = self.to_tensor(reward)

            # Log episode rewards and lengths if available
            for item in info:
                if "episode" in item.keys():
                    self.writer.add_scalar("episode/episodic_return", item["episode"]["r"], self.global_step)
                    self.writer.add_scalar("episode/episodic_length", item["episode"]["l"], self.global_step)

            self.global_step+=1

        self.writer.add_scalar("time/data_conversion", convert_time/self.num_steps, self.reward_step)
        self.writer.add_scalar("time/env_interaction", env_interaction_time/self.num_steps, self.reward_step)
        # Get value for the final observation
        _, v = self.model(self.obs)
        values[:,self.num_steps] = v.reshape(self.n_envs)
        
        advantages = self.GAE(values, rewards, dones)

        if self.policy in self.exception:
            observations = {key: observations[key].reshape(self.batch_size, *observations[key].shape[2:]) for key in observations}
        else: 
            observations = observations.reshape(self.batch_size, *observations.shape[2:])
        return {
            'observations': observations,
            'actions': actions.reshape(self.batch_size, *actions.shape[2:]),
            'values': values[:,:-1].reshape(self.batch_size, *values.shape[2:]),
            'log_prob': log_probs.reshape(self.batch_size, *log_probs.shape[2:]),
            'advantages': advantages.reshape(self.batch_size, *advantages.shape[2:]),
            'rewards': rewards.reshape(self.batch_size, *advantages.shape[2:])
        }
    

    def train(self, samples: dict) -> None:
        """
        Train the model for a fixed number of epochs using mini-batches.

        Args:
            samples (dict): Dictionary containing training samples.
        """
        for _ in range(self.n_epochs):
            idx = torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_idx = idx[start:end]
                if self.policy in self.exception:
                    mini_batch_samples = {
                        k: (
                            {key: v[key][mini_batch_idx].to(self.device) for key in v} if k == "observations" else v[mini_batch_idx].to(self.device)
                        ) 
                        for k, v in samples.items()
                    }
                else:
                    mini_batch_samples = {
                        k: v[mini_batch_idx].to(self.device) for k,v in samples.items()
                    }

                loss = self.compute_loss(mini_batch_samples)
                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                # Calculate and log gradient norm
                total_norm = sum(
                    param.grad.data.norm(2).item() ** 2 for param in self.model.parameters() if param.grad is not None
                ) ** 0.5
                self.writer.add_scalar(
                    "grad_norm",
                    total_norm, 
                    global_step = self.loss_step-1
                )


                self.optimizer.step()


    def compute_loss(self, samples: dict) -> torch.Tensor:
        """
        Compute the PPO loss for a mini-batch.

        Args:
            samples (dict): Mini-batch samples.

        Returns:
            torch.Tensor: Computed loss.
        """
        sample_ret = samples["values"]+samples["advantages"]
        old_values = samples["values"]
        action_distribution,values = self.model(samples["observations"])
        values = values.squeeze(1)
        log_probs = action_distribution.log_prob(samples["actions"])
        adv_norm = normalize(samples["advantages"])

        value_f = (sample_ret-values)**2
        value_pred_clipped = (
            torch.clamp(
                values-old_values, 
                -self.clip_eps, 
                self.clip_eps
            ) + old_values
        )
        value_f_clipped = (value_pred_clipped - sample_ret)**2

        loss = (
            -self.ppo_clip(log_probs, samples["log_prob"], adv_norm).mean()
            + self.c1 * 0.5 * (torch.max(value_f, value_f_clipped)).mean()
            - self.c2 * action_distribution.entropy().mean() 
        )
        self.writer.add_scalar("global_loss",loss, global_step = self.loss_step)
        self.writer.add_scalar(
            "policy_loss",
            self.ppo_clip(log_probs, samples["log_prob"], adv_norm).mean(), 
            global_step = self.loss_step
        )
        self.writer.add_scalar(
            "value_loss",
            ((sample_ret-values)**2).mean(), 
            global_step = self.loss_step
        )
        self.writer.add_scalar(
            "entropy_loss",
            action_distribution.entropy().mean() , 
            global_step = self.loss_step
        )
        self.loss_step+=1

        return loss

    
    def GAE(self, values: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute the Generalized Advantage Estimation (GAE).

        Args:
            values (torch.Tensor): Estimated state values.
            rewards (torch.Tensor): Rewards collected.
            dones (torch.Tensor): Binary flags indicating episode termination.

        Returns:
            torch.Tensor: Computed advantages.
        """

        advantages = torch.zeros_like(rewards)
        last_advantages = 0
        for t in reversed(range(self.num_steps)):
            delta = rewards[:,t] + self.gamma * values[:,t+1] * (1.0 - dones[:,t]) - values[:,t]
            advantages[:,t] = last_advantages = delta + self.gamma * self.gae_lambda * (1.0 - dones[:,t]) * last_advantages
        return advantages
    
    def ppo_clip(self,log_prob: torch.Tensor, log_prob_old: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        Calculate the PPO clipped objective.

        Args:
            log_prob (torch.Tensor): New log probabilities.
            log_prob_old (torch.Tensor): Old log probabilities.
            advantages (torch.Tensor): Advantage estimates.

        Returns:
            torch.Tensor: Clipped PPO loss.
        """
        ratio = torch.exp(log_prob-log_prob_old)
        loss = ratio * advantages
        loss_clip = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
        return torch.min(loss, loss_clip)
    
    def training_loop(self):
        """
        Main training loop.
        """
        for e in tqdm(range(1,self.update+1)):
            if self.anneal_lr:
                coeff = 1 - (e/self.update)
                self.optimizer.param_groups[0]["lr"] = coeff * self.lr

            start_sample = time.time()
            samples = self.sample()
            sample_time = time.time() - start_sample
            self.writer.add_scalar("time/sample_total", sample_time/self.num_steps, self.reward_step)
            self.writer.add_scalar(
                "mean_reward",
                samples["rewards"].mean(), 
                global_step = self.reward_step
            )
            start_train = time.time()
            self.train(samples)
            train_time = time.time() - start_train
            self.writer.add_scalar("time/train_total", train_time, self.reward_step)
            self.reward_step += 1

            if e % 50 == 0:
                models_dir = "models"
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)

                model_path = os.path.join(models_dir, self.model_name)
                self.model.save(model_path)

    @torch.no_grad()
    def test_policy(self,make_env, n_eval_episodes: int = 10) -> tuple:
        """
        Evaluate the current policy over several episodes.

        Args:
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.

        Returns:
            tuple: Mean and standard deviation of rewards over the episodes.
        """
        env = make_env(data_path="./dataset/data_test.pickle") if self.policy in self.exception else make_env()
        rewards = [0]*n_eval_episodes
        for n in range(n_eval_episodes):
            observation,_ = env.reset()
            done = False
            while not done:
                if self.policy in self.exception:
                    obs = self.to_tensor_dict(observation, test=True)
                else:
                    obs = self.to_tensor(observation).reshape(1,-1).to(self.device)
                action_distribution, _ = self.model(obs)
                a = torch.argmax(action_distribution.logits).item()
                observation, reward, done, truncated, _ = env.step(a)
                done = done or truncated
                rewards[n] += reward
        return np.mean(rewards), np.std(rewards)


