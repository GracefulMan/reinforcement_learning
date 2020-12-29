import numpy as np
import pandas as pd
import os
class QlearningAlgorithm:
    def __init__(self,  actions, gamma=0.9, epsilon=0.9, lr = 0.05):
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.lr = lr

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def choose_action(self, ob):
        # epsilon greedy
        ob = str(ob)
        self.check_state_exist(ob)
        prob = np.random.uniform()
        if prob < 1 - self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[ob, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )
