import gym
from Q_learning import QlearningAlgorithm
env=gym.make("CartPole-v1")
RL = QlearningAlgorithm(actions=list(range(env.action_space.n)))
observation = env.reset()
done = False
for _ in range(1000):
    observation = env.reset()
    while not done:
        env.render()
        action =RL.choose_action(observation) # your agent here (this takes random actions)
        observation_, reward, done, info = env.step(action)
        RL.learn(observation,action, reward,observation_)
        observation = observation_
    done = False
    print('epoch:',_)
env.close()