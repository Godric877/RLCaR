from cache import CacheEnv
import matplotlib.pyplot as plt
import numpy as np
from algorithms.semi_gradient_sarsa_algorithm import semi_gradient_sarsa
from approximations.linear_approximation import LinearApproximation

def plot_reward(rewards):
    fig, ax = plt.subplots(1)
    ax[0].plot(np.arange(0, len(rewards)), rewards)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("reward")
    plt.show()

env = CacheEnv()
num_episodes = 1
epsilon = 0.1
actions = [0, 1]
gamma = 0.8
alpha = 0.01
L = LinearApproximation(3, 2, alpha)
rewards = semi_gradient_sarsa(env, gamma, alpha, L, epsilon, num_episodes, actions)
plot_reward(rewards)
