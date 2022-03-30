import numpy as np
from approximations.state_value_approximation import StateValueApproximation

def epsilon_greedy(Q:StateValueApproximation, epsilon, state, actions):
    random = np.random.binomial(1, epsilon)
    max_ac = 0
    if random == 0:
        max_q = np.NINF
        for action in actions:
            current_q = Q(state, action)
            if current_q > max_q:
                max_ac = action
                max_q = current_q
    else:
        action_size = len(actions)
        index = np.random.randint(action_size)
        max_ac = actions[index]
    return max_ac

def semi_gradient_sarsa(env, gamma, alpha, Q:StateValueApproximation,
                        epsilon, episodes, actions):
    #episodes = np.arange(num_episode)
    #train, eval = train_test_split(episodes, test_size=0.2)
    bhr_metric = {}
    rewards = {}
    for i in episodes:
        s_current = env.reset(i)
        action = epsilon_greedy(Q, epsilon, s_current, actions)
        done = False
        episode_rewards = []
        while not done:
            s_next, reward, done, info = env.step(action)
            # if action == 0:
            #     print(action)
            episode_rewards.append(reward)
            if done:
                Q.update(alpha, reward, s_current, action)
                bhr_metric[i] = info[2]
            else:
                next_action = epsilon_greedy(Q, epsilon, s_next, actions)
                G = reward + gamma*Q(s_next, next_action)
                Q.update(alpha, G, s_current, action)
            s_current = s_next
            action = next_action
        rewards[i] = episode_rewards
    return rewards, bhr_metric



