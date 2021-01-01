from DQN.dqn import DeepQNetwork
import gym


def main():
    epsilon = 0.7
    gamma = 0.9
    batch_size = 1024
    memory_size = 4096
    epochs = 10000
    total_step = 0
    lr = 0.01
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape[0]
    actions = env.action_space.n
    dqn = DeepQNetwork(
        observation_dims=observation_dim,
        actions=actions,
        epsilon=epsilon,
        gamma=gamma,
        batch_size=batch_size,
        memory_size=memory_size,
        lr=lr
    )
    for epoch in range(epochs):
        observation = env.reset()
        while True:
            env.render()
            action = dqn.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            dqn.store_transition(observation, action, reward, observation_)
            if total_step > memory_size:
                dqn.learn()
            if done:
                break
            observation = observation_
            total_step += 1
        print('epoch:{}'.format(epoch))


if __name__ == "__main__":
    main()

