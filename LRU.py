from cache import CacheEnv
import numpy as np
import matplotlib.pyplot as plt

def plot_reward(rewards):

    plt.plot(np.arange(0, len(rewards)), np.cumsum(rewards))
    plt.xlabel("time")
    plt.ylabel("cumulative reward")
    plt.show()

def run_env(env, index):
    obs = env.reset(index)
    done = False
    rewards = []
    while not done:
        act = 1
        obs, reward, done, info = env.step(act)
        rewards.append(reward)
    return rewards

env = CacheEnv()
total_reward = []
reward = []
for i in range(0, 20, 1):
    reward = run_env(env, i)
    total_reward = total_reward + reward

plot_reward(reward)


