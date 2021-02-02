from PPO.ppo_discreate import PPO
from PPO.utils import plot_learning_curve
import gym
import os # MACOS setting.
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True' # MACOS setting.


# TODO: use MC doesn't work well, need to use TD-based methods.

def main():
    epochs = 800
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape[0]
    actions = env.action_space.n
    agent = PPO(
        input_dims=observation_dim,
        n_actions=actions
    )
    n_steps = 0
    N = 100
    learn_iters = 0
    score_history = []
    for epoch in range(epochs):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            if epoch > epochs - 10:
                env.render()
            action, prob, value = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, value, prob, reward, done)
            score += reward
            n_steps += 1
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if epoch % 10 == 0: print('epoch:{}, score:{:.2f},avg_score:{:.2f}'.format(epoch, score, avg_score))
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, 'MountainCar1.png')


if __name__ == "__main__":
    main()

