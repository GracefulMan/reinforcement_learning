import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List
import numpy as np

class PolicyGradient(nn.Module):
    # TODO: check the network structure and write test environment.
    def __init__(self,
                 observation_dims: int,
                 actions: int,
                 gamma: float = 0.8,
                 lr: float = 1e-3,
                 hidden_dims: int = 128,
                 ) -> None:
        '''
        observation_dims: input feature nums.
        gamma: reward discount factor, default:0.8
        lr: learning rate, default: 0.001
        '''
        super(PolicyGradient, self).__init__()
        self.gamma = gamma
        self.lr = lr
        self.fc = nn.Sequential(
            nn.Linear(observation_dims, hidden_dims),
            nn.Tanh()
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dims, actions),
            nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, s: torch.FloatTensor) -> Categorical:
        '''
        s: observation
        return: categorical for sample.
        '''
        logits = self.out(self.fc(s))
        return Categorical(logits=logits)

    def choose_action(self, observation: torch.FloatTensor) -> int:
        '''
        observation: the agent observation of current time.
        '''
        action = self.forward(torch.FloatTensor(observation)).sample().item()
        return action

    def _loss_func(self,
                   observations: List[np.ndarray],
                   actions: List[int],
                   rewards: List[float]) -> torch.FloatTensor:
        '''
        the observations, actions and rewards are corresponding to obs, action, reward for one trajectory.
        return:expection of -J(theta), since it's gradient descent, use minus.
        '''
        discount_rewards = []
        tmp = 0
        # calculate accumulate discount reward.
        for r in reversed(rewards):
            tmp = tmp * self.gamma + r
            discount_rewards = [tmp] + discount_rewards
        discount_rewards = torch.FloatTensor(discount_rewards)
        observations = torch.FloatTensor(observations)
        logp = self.forward(observations).log_prob(actions)
        return - (logp * discount_rewards).mean()

    def learn(self,
              observations: List[np.ndarray],
              actions: List[int],
              rewards: List[float],
              ) -> None:
        '''
        study the network params.
        the observations, actions and rewards are corresponding to obs, action, reward for one trajectory.
        '''
        discount_rewards = []
        tmp = 0
        # calculate accumulate discount reward.
        for r in reversed(rewards):
            tmp = tmp * self.gamma + r
            discount_rewards = [tmp] + discount_rewards
        discount_rewards = torch.FloatTensor(discount_rewards)
        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        self.optimizer.zero_grad()
        batch_loss = self._loss_func(observations, actions, discount_rewards)
        batch_loss.backward()
        self.optimizer.step()

















