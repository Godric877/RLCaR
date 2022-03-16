from cache import CacheEnv
import numpy as np
import matplotlib.pyplot as plt

def run_env(env_name, seed):
    env = CacheEnv()
    env.seed(seed)

    obs = env.reset()
    done = False
    i = 0
    count_unique = 0
    rewards = []
    while not done:
        # act = np.random.randint(2) random policy
        act = 1
        obs, reward, done, info = env.step(act)
        #print("Step = ",i)
        # print(obs)
        # print(reward)
        # print(info)
        if obs[2] != 500:
            print(obs)
        else:
            count_unique += 1
        i = i + 1
        rewards.append(reward)

    print(count_unique)
    print("Steps  = ",i)

    # fig, ax = plt.subplots(2)
    # ax[0].plot(np.arange(0, i), rewards)
    # ax[0].set_xlabel("Time")
    # ax[0].set_ylabel("reward")
    #
    # ax[1].plot(np.arange(0, i), np.cumsum(rewards))
    # ax[1].set_xlabel("Time")
    # ax[1].set_ylabel("cumulative reward")
    # plt.show()

run_env('cache', 1)
