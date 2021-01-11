'''
implementation of Actor Critic model by PyTorch.
Author: GracefulMan
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Union
import copy

class GenericNet(nn.Module):
    '''
    generic network used by actor and critic network.
    '''
    def _conv(self, in_channels: int, out_channels: int, kernel_size:Union[int, tuple]) -> nn.Module:
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return layer

    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 lr: float = 1e-4,
                 fc1_dims: int = 32,
                 fc2_dims: int = 16
                 ) -> None:
        super(GenericNet, self).__init__()
        self.lr = lr
        self.conv1 = self._conv(3, 8, 5)
        self.conv2 = self._conv(8, 16, 5)
        self.conv3 = self._conv(16, 32, 5)
        self.conv4 = self._conv(32, 64, 5)
        self.conv_layers = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3456, fc1_dims),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU()
        )
        # critic network's output is a critic value. don't want to use softmax.
        self.fc3 = nn.Linear(fc2_dims, output_dims)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.reshape((x.shape[2], x.shape[0], x.shape[1]))
        x = torch.Tensor(x[None, :]).to(self.device)
        x = self.conv_layers(x)
        x = x.view(x.size()[0], -1)
        return self.fc3(self.fc2(self.fc1(x)))


class ActorCriticNet:
    def __init__(self, observation_dims: int,
                 actions: int,
                 gamma: float = 0.9,
                 actor_lr: float = 1e-4,
                 critic_lr: int = 1e-3) -> None:
        self.actions = actions
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor = GenericNet(input_dims=observation_dims, output_dims=actions, lr=self.actor_lr)
        self.critic = GenericNet(input_dims=observation_dims, output_dims=1, lr=self.critic_lr)
        self.log_probs = None

    def choose_action(self, observation: np.ndarray) -> int:
        '''
        choose action for current state.
        observation: current state
        return: action, type: int
        '''
        probablities =F.softmax(self.actor.forward(observation))
        action_probs = torch.distributions.Categorical(probablities)
        action = action_probs.sample()
        self.log_probs =action_probs.log_prob(action)
        return action.item()

    def _loss_func(self, observation: np.ndarray, reward: float, observation_: np.ndarray, done: bool) -> torch.Tensor:
        critic_value = self.critic.forward(observation)
        critic_value_ = self.critic.forward(observation_)
        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value
        actor_loss = - self.log_probs * delta
        critic_loss = delta ** 2
        loss = actor_loss + critic_loss
        return loss

    def learn(self, observation: np.ndarray,  reward: float, observation_: np.ndarray, done: bool) -> None:
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        loss = self._loss_func(observation, reward, observation_, done)
        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
