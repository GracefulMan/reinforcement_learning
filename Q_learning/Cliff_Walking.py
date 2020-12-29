import  numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy


class CliffWalking_env:

    def __init__(self):
        self.map = np.zeros((4, 12))
        self.map[3, 1:-1] = 1
        self.current_pos = [3, 0]
        self.terminal_pos = [3, 11]
        self.terminal = False

    def actions(self):
        return list(range(4))

    def reset(self):
        self.current_pos = [3, 0]
        self.terminal = False
        return copy.deepcopy(self.current_pos)

    def step(self, action):
        old_pos = copy.deepcopy(self.current_pos)
        reward = -0.1
        if action == 0:
            if self.current_pos[0] > 0:
                self.current_pos[0] -= 1
            else:
                reward += -0.5
        if action == 1:
            if self.current_pos[0] < self.map.shape[0]-1:
                self.current_pos[0] += 1
            else:
                reward += -0.5
        if action == 2:
            if self.current_pos[1] > 0:
                self.current_pos[1] -= 1
            else:
                reward += -0.5
        if action == 3:
            if self.current_pos[1] < self.map.shape[1] - 1:
                self.current_pos[1] += 1
            else:
                reward += -0.5

        if self.map[self.current_pos[0], self.current_pos[1]] == 1:
            reward += -10
            self.terminal = True

        if self.current_pos == self.terminal_pos:
            self.terminal = True
            reward = 10
        return old_pos, reward, self.terminal, 0 # s,r, T,info
