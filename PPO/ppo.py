'''
implementaion of PPO2
'''
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


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


class ActorNetwork(nn.Module):
    def __init__(self,  input_dims: int, n_actions: int, lr:float=1e-3,device: str='cpu') -> None:
        super(ActorNetwork, self).__init__()
        self.hidden = 128
        self.lr = lr
        self.actor = nn.Sequential(
            nn.Linear(input_dims, self.hidden * 2),
            nn.ReLU(),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, n_actions),
            nn.Softmax(dim=1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(device=torch.device(device=device))

    def forward(self, x) -> torch.distributions.Categorical:
        x = self.actor(x)
        dist = torch.distributions.Categorical(x)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims: int, lr: float=1e-4, device: str='cpu' ) -> None:
        super(CriticNetwork, self).__init__()
        self.lr = lr
        self.hidden = 128
        self.critic = nn.Sequential(
            nn.Linear(input_dims, self.hidden * 2),
            nn.ReLU(),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(device=torch.device(device=device))

    def forward(self, x) -> torch.Tensor:
        return self.critic(x)


class PPO:
    def __init__(self,
                 input_dims: int,
                 n_actions: int,
                 batch_size: int = 64,
                 policy_clip: float = 0.2,
                 gamma: float = 0.9,
                 lr: float = 1e-3,
                 gae_lambda: float = 0.95,
                 n_epochs: int = 10
                 ) -> None:
        self.batch_size = batch_size
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.actor = ActorNetwork(input_dims=input_dims, n_actions=n_actions, lr=lr, device=self.device)
        self.critic = CriticNetwork(input_dims=input_dims, lr=lr, device=self.device)
        self.memory = PPOMemory(batch_size=self.batch_size)

    def choose_action(self, observation: np.ndarray) -> Tuple[int, float, float]:
        observation = torch.Tensor([observation]).to(torch.device(self.device))
        dist = self.actor(observation)
        value = self.critic(observation)
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
            advantage = torch.Tensor(advantage).to(torch.device(self.device))
            values = torch.Tensor(values).to(torch.device(self.device))
            for batch in batches:
                observations = torch.Tensor(observation_arr[batch]).to(torch.device(self.device))
                old_probs = torch.Tensor(old_prob_arr[batch]).to(torch.device(self.device))
                actions = torch.Tensor(action_arr[batch]).to(torch.device(self.device))
                dist = self.actor(observations)
                new_probs = dist.log_prob(actions)
                critic_value = torch.squeeze(self.critic(observations))
                prob_ratio = torch.exp(new_probs - old_probs)
                weighted_probs = prob_ratio * advantage[batch]
                cliped_weighted_probs = torch.clip(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = - torch.min(weighted_probs, cliped_weighted_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = torch.mean((returns - critic_value) ** 2)
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()