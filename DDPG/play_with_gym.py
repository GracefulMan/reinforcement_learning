from DDPG.ddpg import DDPG
from DDPG.utils import plot_learning_curve
import gym
import os # MACOS setting.
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True' # MACOS setting.



def main():
    epochs = 1000
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPG(input_dims=8, n_actions=2, tau=0.001, batch_size=64, lr=1e-4)
    score_history = []
    for epoch in range(epochs):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            score += reward
            agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if epoch % 10 == 0: print('epoch:{}, score:{:.2f},avg_score:{:.2f}'.format(epoch, score, avg_score))
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, 'LunarLanderContinuous.png')


if __name__ == "__main__":
    main()

