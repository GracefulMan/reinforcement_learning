from Policy_Optimization.policy_gradient import PolicyGradient
import gym
import os # MACOS setting.
os.environ['KMP_DUPLICATE_LIB_OK']='True' # MACOS setting.


# TODO: use MC doesn't work well, need to use TD-based methods.

def main():
    gamma = 0.9
    epochs = 10000
    lr = 0.01
    env = gym.make('MountainCar-v0')
    observation_dim = env.observation_space.shape[0]
    actions = env.action_space.n
    agent = PolicyGradient(
        observation_dims=observation_dim,
        actions=actions,
        gamma=gamma,
        lr=lr
    )
    for epoch in range(epochs):
        observation = env.reset()
        obs = []
        actions = []
        rewards = []
        while True:
            if epoch > 1000:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            obs.append(observation)
            actions.append(action)
            rewards.append(reward)
            if done:
                agent.learn(obs, actions, rewards)
                break
            observation = observation_
        print('current epoch:{}'.format(epoch))


if __name__ == "__main__":
    main()

