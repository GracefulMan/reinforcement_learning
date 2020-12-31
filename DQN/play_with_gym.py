from DQN.dqn import DeepQNetwork
import gym

def main():
    epsilon = 0.8
    gamma = 0.9
    batch_size = 64
    memory_size = 2048
    epochs = 1000
    total_step = 0
    env = gym.make('CartPole-v1')
    observation_dim = env.observation_space.shape[0]
    actions = env.action_space.n
    dqn = DeepQNetwork(
        observation_dims=observation_dim,
        actions=actions,
        epsilon=epsilon,
        gamma=gamma,
        batch_size=batch_size,
        memory_size=memory_size
    )
    for epoch in range(epochs):
        observation = env.reset()
        while True:
            env.render()
            action = dqn.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            dqn.store_transition(observation, action, reward, observation_)
            if total_step > 1000:
                dqn.learn()
            if done:
                break
            observation = observation_
        print('epoch:{}'.format(epoch))





if __name__ == "__main__":
    main()

