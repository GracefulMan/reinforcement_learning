import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Net(nn.Module):
    def __init__(self, observation_dims: int, actions: int, hidden_dims=128) -> None:
        super(Net, self).__init__()
        self.n_actions = actions
        self.fc = nn.Sequential(
            nn.Linear(in_features=observation_dims, out_features = hidden_dims),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_dims, out_features=actions),
            nn.Softmax(dim=1)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.out(self.fc(s))


class DeepQNetwork:
    def __init__(self,
                 observation_dims: int,
                 actions: int,
                 hidden_dims=128,
                 memory_size=2048,
                 lr=1e-4,
                 epsilon=0.8,
                 gamma=0.8,
                 q_iteration=100,
                 batch_size=64

                 ) -> None:
        super(DeepQNetwork, self).__init__()
        self.q_target = Net(observation_dims=observation_dims, actions=actions,hidden_dims=hidden_dims)
        self.q_eval = Net(observation_dims=observation_dims, actions=actions,hidden_dims=hidden_dims)
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.memory = np.zeros((memory_size, observation_dims * 2 + 2))
        self.optimizer = Adam(self.q_eval.parameters(), lr=lr)
        self.loss_func = MSELoss()

        self.epsilon = epsilon # epsilon greedy params
        self.gamma = gamma
        self.actions = actions
        self.n_states = observation_dims
        self.memory_size = memory_size
        self.q_net_iteration = q_iteration
        self.batch_size = batch_size

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.Tensor(observation[np.newaxis, :])
        # using epsilon-greedy algorithm to get the action.
        prob = np.random.uniform()
        if prob < self.epsilon:
            action_prob = self.q_eval.forward(observation)
            action = torch.argmax(action_prob).tolist()
        else:
            action = int(np.random.randint(0, self.actions))
        return action

    def learn(self):
        # update the parameters
        if self.learn_step_counter % self.q_net_iteration == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        self.learn_step_counter += 1
        # sample batch from memory
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.n_states])
        bation_action = torch.LongTensor(batch_memory[:, self.n_states:self.n_states+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.n_states+1:self.n_states+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.n_states:])
        # q_eval
        q_eval = self.q_eval(batch_state).gather(1, bation_action)
        q_next = self.q_target(batch_next_state).detach() # don't upate the params of q target network.
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


