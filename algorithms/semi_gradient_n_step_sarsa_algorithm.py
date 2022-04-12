# One step Semi Gradient Sarsa

import numpy as np
import math
import copy

from state_action_approximations.state_action_approximation import StateActionApproximation

def epsilon_greedy(Q:StateActionApproximation, epsilon, state, actions):
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

def semi_gradient_n_step_sarsa(env, gamma, alpha, Q:StateActionApproximation,
                               epsilon, episodes, actions, n):
    bhr_metric = {}
    rewards = {}
    for i in episodes:
        s_current = env.reset(i)
        action = epsilon_greedy(Q, epsilon, s_current, actions)
        done = False
        episode_rewards = []
        episode = []
        t = 0
        T = math.inf
        while not done:
            if(t < T):
                s_next, reward, done, info = env.step(action)
                episode.append((s_current, action, s_next, reward))
                episode_rewards.append(reward)
                if done:
                    T = t + 1
                    bhr_metric[i] = info[2]
                else:
                    next_action = epsilon_greedy(Q, epsilon, s_next, actions)
                s_current = s_next
                action = next_action
            tau = t - n + 1
            if(tau >= 0):
                G = 0
                discount = 1
                for j in range(tau, min(tau+n, T)):
                    G += discount*episode[j][3]
                    discount *= gamma
                if(tau+n < T):
                    G = G + np.power(gamma,n)*Q(episode[tau+n-1][0], episode[tau+n-1][1])
                Q.update(alpha, G, episode[tau][0], episode[tau][1])
            t += 1
            if(tau == T-1):
                break
        rewards[i] = episode_rewards
    return rewards, bhr_metric



