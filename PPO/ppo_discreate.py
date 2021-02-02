import torch
import torch.nn as nn
import os
from typing import Tuple
import numpy as np
'''
implement PPO algorithm for my navigation.
'''


class PPOMemory:
    '''
    store the interact data.
    '''
    def __init__(self, batch_size: int = 32) -> None:
        self.observations = []
        self.actions = []
        self.vals = [] # value of state.
        self.probs = [] # log probs.
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self) -> Tuple:
        n_states = len(self.observations)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        return np.array(self.observations), np.array(self.actions), np.array(self.vals),\
               np.array(self.probs), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self,
                     observation: np.ndarray,
                     action: int,
                     val: float,
                     prob: float,
                     rewards: float,
                     done: bool
                     ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.vals.append(val)
        self.probs.append(prob)
        self.rewards.append(rewards)
        self.dones.append(done)

    def clear_memory(self) -> None:
        self.observations = []
        self.actions = []
        self.vals = []
        self.probs = []
        self.rewards = []
        self.dones = []


class ActorCriticNetwork(nn.Module):

    def __init__(self, input_dims: int, n_actions: int, weight_name: str = 'ppo', hidden_dim=512,
                 lr: float = 1e-3) -> None:
        super(ActorCriticNetwork, self).__init__()
        self.lr = lr
        self.weight_name = weight_name
        self.root_path = 'weights/'  # save model weights
        self.fc = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(int(hidden_dim / 4), n_actions),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Linear(int(hidden_dim / 4), 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.load_model_weight()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def save_model_weight(self) -> None:
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        weight_path = os.path.join(self.root_path, self.weight_name) + '.weight'
        torch.save(self.state_dict(), weight_path)

    def load_model_weight(self) -> None:
        weight_path = os.path.join(self.root_path, self.weight_name) + '.weight'
        if os.path.exists(weight_path):
            self.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        x = self.fc(x)
        dist = torch.distributions.Categorical(self.actor(x))
        value = self.critic(x)
        return dist, value



class PPO:
    def __init__(self,
                 input_dims: int,
                 n_actions: int,
                 batch_size: int = 128,
                 policy_clip: float = 0.2,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 gae_lambda: float = 0.95,
                 n_epochs: int = 20
                 ) -> None:
        self.batch_size = batch_size
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.ActorCritic = ActorCriticNetwork(input_dims, n_actions, lr=lr)
        self.memory = PPOMemory(batch_size=self.batch_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def choose_action(self, observation: np.ndarray) -> Tuple[int, float, float]:
        observation = torch.Tensor([observation]).to(self.device)
        dist, value = self.ActorCritic(observation)
        action = dist.sample()
        prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, prob, value

    def remember(self,
                 observation: np.ndarray,
                 action: int,
                 val: float,
                 prob: float,
                 rewards: float,
                 done: bool) -> None:
        self.memory.store_memory(observation=observation, action=action, val=val, prob=prob, rewards=rewards, done=done)

    def learn(self) -> None:
        for _ in range(self.n_epochs):
            observation_arr, action_arr, val_arr, old_prob_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
            values = val_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1 - int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.Tensor(advantage).to(torch.device(self.device))
            values = torch.Tensor(values).to(torch.device(self.device))
            for batch in batches:
                observations = torch.Tensor(observation_arr[batch]).to(torch.device(self.device))
                old_probs = torch.Tensor(old_prob_arr[batch]).to(torch.device(self.device))
                actions = torch.Tensor(action_arr[batch]).to(torch.device(self.device))
                dist, value = self.ActorCritic(observations)
                new_probs = dist.log_prob(actions)
                critic_value = torch.squeeze(value)
                prob_ratio = torch.exp(new_probs - old_probs)
                weighted_probs = prob_ratio * advantage[batch]
                cliped_weighted_probs = torch.clip(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = - torch.min(weighted_probs, cliped_weighted_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = torch.mean((returns - critic_value) ** 2)
                total_loss = actor_loss + 0.5 * critic_loss
                self.ActorCritic.optimizer.zero_grad()
                total_loss.backward()
                self.ActorCritic.optimizer.step()

        self.memory.clear_memory()



