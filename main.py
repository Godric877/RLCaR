from cache import CacheEnv
import matplotlib.pyplot as plt
import numpy as np
from algorithms.semi_gradient_sarsa_algorithm import semi_gradient_sarsa
from algorithms.LRU import lru
from approximations.linear_approximation import LinearApproximation
from sklearn.model_selection import train_test_split


def plot_reward(rewards, filename):
    plt.plot(np.arange(0, len(rewards)), np.cumsum(rewards))
    plt.xlabel("time")
    plt.ylabel("cumulative reward")
    plt.savefig(filename)
    plt.close()
    #plt.show()

def get_metrics(test, episode, rewards, bhr, filename):
    avg_test_bhr = 0.0
    for index in test:
        avg_test_bhr += bhr[index]
    avg_test_bhr /= len(test)
    print("Average BHR on test data : ", avg_test_bhr)
    plot_reward(rewards[test[episode]], filename)

def semi_gradient_sarsa_with_linear_approx(env, train, test):
    # Semi-gradient Sarsa with linear function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha = 0.01
    L = LinearApproximation(3, 2, alpha)
    semi_gradient_sarsa(env, gamma, alpha, L, epsilon, train, actions)
    rewards, bhr = semi_gradient_sarsa(env, gamma, alpha, L, epsilon, test, actions)
    get_metrics(test, 0, rewards, bhr, "experiments/graphs/semi_gradient_sarsa.png")

def run_lru(env, train, test):
    lru(env, train)
    rewards, bhr = lru(env, test)
    get_metrics(test, 0, rewards, bhr, "experiments/graphs/lru.png")

if __name__ == "__main__":
    env = CacheEnv()
    num_episodes = 20
    episodes = np.arange(num_episodes)
    train, test = train_test_split(episodes, test_size=0.2)
    #run_lru(env, train, test)
    semi_gradient_sarsa_with_linear_approx(env, train, test)









