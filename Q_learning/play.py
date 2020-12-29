from Cliff_Walking import CliffWalking_env
from Q_learning import QlearningAlgorithm

def run():
    env = CliffWalking_env()
    actions = env.actions()
    RL = QlearningAlgorithm(actions)
    epochs = 1000
    for epoch in range(epochs):
        observation = env.reset()
        while True:
            action = RL.choose_action(str(observation))
            s_, r, terminal, _ = env.step(action)
            if terminal:
                s_ = 'terminal'
            RL.learn(str(observation), action, r, str(s_))
            observation = s_
            if terminal:
                break
        print('epoch:{}'.format(epoch))


if __name__ == "__main__":
    run()


