'''
implementation of Actor Critic model by PyTorch.
Author: GracefulMan
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple


class ActorCriticNet(nn.Module):
    def _conv(self, in_channels: int, out_channels: int) -> nn.Module:
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        return layer

    def __init__(self,observation_dims: int, actions: int, gamma: float=0.99, lr: float=0.01) -> None:
        super(ActorCriticNet, self).__init__()
        self.gamma = gamma
        self.lr = lr
        self.conv = nn.Sequential(
            self._conv(observation_dims, 8),
            self._conv(8, 16),
            self._conv(16, 32),
        )
        self.fc1 = nn.Linear(11264, actions)
        self.fc2 = nn.Linear(11264 , 1)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)
        self.optimizer = torch.optim.RMSprop(self.parameters(), self.lr)

    def forward(self, observation: np.ndarray) -> Tuple[int, float]:
        x = observation.reshape((observation.shape[2], observation.shape[0], observation.shape[1]))
        x = torch.Tensor(x[None,:]).to(self.device)
        x = self.conv(x)
        x = x.view(x.size()[0],-1)
        probablities = F.softmax(self.fc1(x))
        value = self.fc2(x)
        return probablities, value

    def choose_action(self, observation:np.ndarray) -> int:
        probablities, _ = self.forward(observation)
        action_probs = torch.distributions.Categorical(probablities)
        action = action_probs.sample().item()
        return action

    def _loss_func(self, observation: np.ndarray, reward: float, observation_: np.ndarray, done: bool) -> torch.Tensor:
            probablities, critic_value = self.forward(observation)
            action_probs = torch.distributions.Categorical(probablities)
            action = action_probs.sample()
            log_probs = action_probs.log_prob(action)
            _, critic_value_ = self.forward(observation_)
            delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value
            actor_loss = - log_probs * delta
            critic_loss = delta ** 2
            loss = actor_loss + critic_loss
            return loss

    def learn(self, observation: np.ndarray,  reward: float, observation_: np.ndarray, done: bool) -> None:
        self.optimizer.zero_grad()
        loss = self._loss_func(observation, reward, observation_, done)
        loss.backward()
        self.optimizer.step()
