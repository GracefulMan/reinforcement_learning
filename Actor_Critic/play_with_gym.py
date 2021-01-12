from Actor_Critic.model_v2 import ActorCriticNet
import gym
import matplotlib.pyplot as plt
import numpy as np

def plot_reward(reward_history: list, window: int=5) -> None:
    N = len(reward_history)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(reward_history[max(0, t - window): (t + 1)])
    plt.plot(list(range(len(reward_history))), running_avg)
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.show()


def main():
    gamma = 0.9
    epochs = 1000
    env = gym.make('MsPacman-v0')
    observation_dim = env.observation_space.shape
    actions = env.action_space.n
    Agent = ActorCriticNet(observation_dims=observation_dim[2],actions=actions, gamma=gamma)
    score_history = []
    for epoch in range(epochs):
        score = 0
        observation = env.reset()
        while True:
            if epoch > epochs - 50:
                env.render()
            action = Agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            Agent.learn(observation, reward, observation_, done)
            if done:
                break
            observation = observation_
        score_history.append(score)
        print('epoch:{},score:{:.2f}'.format(epoch, score))
    plot_reward(reward_history=score_history)


if __name__ == "__main__":
    main()

