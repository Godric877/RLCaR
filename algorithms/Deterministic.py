import numpy as np

def always_evict(env, episodes):
    bhr_metric = {}
    rewards = {}
    for i in episodes:
        env.reset(i)
        done = False
        episode_rewards = []
        while not done:
            act = 1
            obs, reward, done, info = env.step(act)
            episode_rewards.append(reward)
            if done:
                bhr_metric[i] = info[2]
        rewards[i] =  episode_rewards
    return rewards, bhr_metric

def random_eviction(env, episodes):
    bhr_metric = {}
    rewards = {}
    for i in episodes:
        env.reset(i)
        done = False
        episode_rewards = []
        while not done:
            act = np.random.choice(np.arange(2), p=[0.8, 0.2])
            obs, reward, done, info = env.step(act)
            episode_rewards.append(reward)
            if done:
                bhr_metric[i] = info[2]
        rewards[i] =  episode_rewards
    return rewards, bhr_metric





