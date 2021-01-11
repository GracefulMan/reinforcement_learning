from Actor_Critic.model import ActorCriticNet
import gym
import matplotlib.pyplot as plt


def plot_reward(reward_history: list) -> None:
    plt.plot(list(range(len(reward_history))), reward_history)
    plt.show()


def main():
    gamma = 0.9
    epochs = 2500
    env = gym.make('CartPole-v1')
    observation_dim = env.observation_space.shape[0]
    actions = env.action_space.n
    Agent = ActorCriticNet(observation_dims=observation_dim,actions=actions,gamma=gamma)
    score_history = []
    for epoch in range(epochs):
        score = 0
        observation = env.reset()
        while True:
            if epoch > 2490:
                env.render()
            action = Agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            Agent.learn(observation,reward,observation_,done)
            if done:
                break
            observation = observation_
        score_history.append(score)
        print('epoch:{}'.format(epoch))
    plot_reward(reward_history=score_history)

if __name__ == "__main__":
    main()

