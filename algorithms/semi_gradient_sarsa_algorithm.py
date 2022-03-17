from cache import CacheEnv
import numpy as np
from approximations.state_value_approximation import StateValueApproximation

def epsilon_greedy(Q:StateValueApproximation, epsilon, state, actions):
    random = np.random.binomial(1, epsilon)
    max_ac = 0
    if random == 1:
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

def semi_gradient_sarsa(env, gamma, alpha, Q:StateValueApproximation, epsilon, num_episode, actions):
    for i in range(0, num_episode, 1):
        s_current = env.reset()
        action = epsilon_greedy(Q, epsilon, s_current, actions)
        done = False
        rewards = []
        while not done:
            s_next, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                Q.update(alpha, reward, s_current, action)
                break
            else:
                next_action = epsilon_greedy(Q, epsilon, s_next, actions)
                G = reward + gamma*Q(s_next, next_action)
                Q.update(alpha, G, s_current, action)
            s_current = s_next
            action = next_action
    return rewards



