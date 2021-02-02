import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from typing import Tuple
import os


class OUActionNoise:
    def __init__(self, mu:np.ndarray, sigma: float=0.15, theta: float=0.2, dt: float=1e-2, x0:np.ndarray=None) -> None:
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0

    def __call__(self, *args, **kwargs):
        x = self.x_prev + self.theta* (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class ReplayBuffer:
    """
    memory replay buffer.
    """
    def __init__(self, max_size: int, input_shape: int, n_actions: int) -> None:
        """
        @param max_size: maximum size of buffer. (like a queue)
        @param input_shape: the shape of env s_t.
        @param n_actions:  the action size.
        """
        self.max_size = max_size
        self.counter = 0
        self.states_memory = np.zeros((max_size, input_shape))
        self.actions_memory = np.zeros((max_size, n_actions))
        self.reward_memory = np.zeros(max_size)
        self.states_memory_ = np.zeros((max_size, input_shape))
        self.terminal_memory = np.zeros(max_size)

    def store_transition(self, state: np.ndarray, action: float, reward: float, state_: np.ndarray, done: bool) -> None:
        """
        @param state: s_t
        @param action: a_t
        @param reward: r_t
        @param state_: s_{t+1}
        @param done: bool value, '1' indicate terminal and '0' indicate doesn't terminal.
        @return: None.
        """
        index = self.counter % self.max_size
        self.states_memory[index] = state
        self.actions_memory[index] = action
        self.reward_memory[index] = reward
        self.states_memory_[index] = state_
        self.terminal_memory[index] = done

    def sample_buffer(self, batch_size: int) -> Tuple:
        """
        @param batch_size: the number of data to sample.
        @return: List[Tuple]->[(s_t, a_t, r_t, s_{t+1}, done),(...)].
        """
        max_size = min(self.counter, self.max_size)
        index = np.random.choice(max_size, batch_size)
        states = self.states_memory[index]
        actions = self.actions_memory[index]
        rewards = self.reward_memory[index]
        states_ = self.states_memory_[index]
        done = self.terminal_memory[index]
        return states, actions, rewards, states_, done


class CriticNetwork(nn.Module):
    def __init__(self, input_dims: int, n_actions: int, name: str,lr: float=1e-3, chkpt: str='model/ddpg') -> None:
        super(CriticNetwork, self).__init__()
        self.chkpt_path = os.path.join(chkpt, name + '_ddpg')
        fc1_dim = 128
        fc2_dim = 64
        self.fc1 = self._linear_layer(input_dims, fc1_dim)
        self.fc2 = self._linear_layer(fc1_dim, fc2_dim)
        self.action_value = nn.Linear(n_actions, fc2_dim)
        self.q = self._linear_layer(fc2_dim, 1, clip_val=0.003, bn=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _linear_layer(self, input_dims: int, output_dims:int, clip_val: float=None, bn: bool=True) -> nn.Module:
        layer = nn.Linear(input_dims, output_dims)
        tmp_value = 1 / np.sqrt(layer.weight.data.size()[0]) if clip_val is None else clip_val
        torch.nn.init.uniform_(layer.weight.data, -tmp_value, tmp_value)
        torch.nn.init.uniform_(layer.bias.data, -tmp_value, tmp_value)
        if bn:
            layer = nn.LayerNorm(input_dims)(layer)
        return layer

    def forward(self, state:np.ndarray, action:float) -> torch.Tensor:
        state_value = torch.relu(self.fc1(state))
        state_value = torch.relu(self.fc2(state_value))
        action_value = torch.relu(self.action_value(action))
        state_action_value = torch.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self) -> None:
        print('...save checkpoints...')
        torch.save(self.state_dict(), self.chkpt_path)

    def load_checkpoint(self) -> None:
        print('...load checkpoints...')
        if os.path.exists(self.chkpt_path):
            self.load_state_dict(torch.load(self.chkpt_path))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims:int, n_actions:int, name: str, lr: float=1e-3, chkpt:str='model/ddpg' )-> None:
        super(ActorNetwork, self).__init__()
        fc1_dim = 128
        fc2_dim = 64
        self.fc1 = self._linear_layer(input_dims, fc1_dim)
        self.fc2 = self._linear_layer(fc1_dim, fc2_dim)
        self.mu = self._linear_layer(fc2_dim, n_actions, clip_val=0.003, bn=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _linear_layer(self, input_dims: int, output_dims:int, clip_val: float=None, bn: bool=True) -> nn.Module:
        layer = nn.Linear(input_dims, output_dims)
        tmp_value = 1 / np.sqrt(layer.weight.data.size()[0]) if clip_val is None else clip_val
        torch.nn.init.uniform_(layer.weight.data, -tmp_value, tmp_value)
        torch.nn.init.uniform_(layer.bias.data, -tmp_value, tmp_value)
        if bn:
            layer = nn.LayerNorm(input_dims)(layer)
        return layer

    def forward(self, state:np.ndarray) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.mu(x))
        return x
    def save_checkpoint(self) -> None:
        print('...save checkpoints...')
        torch.save(self.state_dict(), self.chkpt_path)

    def load_checkpoint(self) -> None:
        print('...load checkpoints...')
        if os.path.exists(self.chkpt_path):
            self.load_state_dict(torch.load(self.chkpt_path))


class DDPG:
    def __init__(self, input_dims:int, n_actions:int ,tau, env, lr:float=1e-3, gamma: float=0.99, max_buffer_size=1000000, batch_size=64) -> None:
        self.gamma = gamma
        self.tau =tau
        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=input_dims, n_actions=n_actions)
        self.batch_size =batch_size
        self.actor = ActorNetwork(input_dims, n_actions, 'Actor', lr=lr)
        self.target_actor = ActorNetwork(input_dims, n_actions, 'TargetActor', lr=lr)
        self.critic = CriticNetwork(input_dims,n_actions,'Critic',lr=lr)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1)


