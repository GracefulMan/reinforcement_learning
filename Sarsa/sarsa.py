import numpy as np
import pandas as pd

class SarsaAlgorithm:
    '''
    implement of Sarsa algorithm.
    '''
    def __init__(self,actions:list, gamma=0.9, epsilon=0.8, lr=0.05):
        self.q_table = pd.DataFrame(columns=actions,dtype=np.float64)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.actions = actions


    def check_state_exist(self, s):
        '''
        check whether state s exist in Q table.
        s:observation  s_t
        '''
        if s not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=s
                )
            )

    def choose_action(self, s):
        # use epsilon-greedy algorithm to get the action.
        s = str(s) # convert observation to string type.
        self.check_state_exist(s)
        prob = np.random.uniform()
        if prob < self.epsilon:
            state_action = self.q_table.loc[s, :]
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, a_):
        s_ = str(s_)
        s = str(s)
        self.check_state_exist(s_)
        # get Q(s_, a')
        q_predict = self.q_table.loc[s, a]
        Q_s_hat_a_hat = self.q_table.loc[s_, a_]
        if s_ == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * Q_s_hat_a_hat
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

