def lru(env, episodes):
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





