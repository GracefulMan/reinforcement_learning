import numpy as np
# 状态转移矩阵P(s'| s, a)
P = np.array([
    [0, 1, 0],
    [0.02, 0.17, 0.81],
    [0, 0.8, 0.2]
])
# R(s, a)
R = np.array([
    [1000, 0],
    [100, -100],
    [-50, 0]
])

# 贝尔曼迭代 V_t = R + \gamma * P * V_{t+1}
gamma = 0.9 # discount factor
V = np.linalg.pinv(np.eye(P.shape[0]) -gamma * P ) @ R
print(np.round(V, 2))

index = np.argmax(V,axis=1)
print(index)