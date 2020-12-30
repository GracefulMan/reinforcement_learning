import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

class DeepQNetwork(nn.Module):
    def __init__(self, observation_dims: int, actions: int, hidden_dims=128, epsilon = 0.8) -> None:
        super(DeepQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=observation_dims, out_features = hidden_dims),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_dims, out_features=actions),
            nn.Softmax(),
        )

    def forward(self, s: np.ndarray) -> torch.Tensor:
        return self.out(self.fc(s))

Q_net = DeepQNetwork(observation_dims=3, actions=4)

# TODO: implement DeepQNetwork with gym environment.
