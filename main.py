from cache import CacheEnv
import matplotlib.pyplot as plt
import numpy as np
from algorithms.semi_gradient_sarsa_algorithm import semi_gradient_sarsa
from approximations.linear_approximation import LinearApproximation

def plot_reward(rewards):

    plt.plot(np.arange(0, len(rewards)), np.cumsum(rewards))
    plt.xlabel("time")
    plt.ylabel("cumulative reward")
    plt.show()

env = CacheEnv()
num_episodes = 20
epsilon = 0.1
actions = [0, 1]
gamma = 1
alpha = 0.01
L = LinearApproximation(3, 1, alpha)
rewards, bhr = semi_gradient_sarsa(env, gamma, alpha, L, epsilon, num_episodes, actions)
print(bhr.mean())
plot_reward(rewards)
