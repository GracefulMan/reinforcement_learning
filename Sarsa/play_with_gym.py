import gym
from Sarsa.sarsa import SarsaAlgorithm
env=gym.make("CartPole-v1")
RL = SarsaAlgorithm(actions=list(range(env.action_space.n)))
done = False
for _ in range(10000):
    s = env.reset()
    while not done:
        env.render()
        a =RL.choose_action(s) # your agent here (this takes random actions)
        s_, r, done, info = env.step(a)
        a_ = RL.choose_action(s_)
        RL.learn(s, a, r, s_, a_)
        s = s_
    done = False
    print('epoch:', _)
env.close()