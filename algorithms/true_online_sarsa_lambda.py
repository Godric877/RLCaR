import copy
import math
import numpy as np

from state_action_approximations.one_d_tc import StateActionOneDTileCoding

def TrueOnlineSarsaLambda(
    env,
    epsilon:float, # exploration factor
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionOneDTileCoding,
    episodes,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,a,done)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    bhr_metric = {}
    rewards = {}
    #TODO: implement this function
    for i in episodes:
        state = env.reset(i)
        done = False
        eps = copy.deepcopy(epsilon)
        action = epsilon_greedy_policy(state, done, w, eps)
        q_old = 0
        x = X(state, action, done)
        z = np.zeros(shape=X.feature_vector_len())
        t = 1
        episode_rewards = []
        while not done:
            t+=1
            state_new, r, done, info = env.step(action)
            episode_rewards.append(r)
            action_new = epsilon_greedy_policy(state_new, done, w, epsilon)
            x_new = X(state_new, action_new, done)
            q = np.dot(w, x)
            q_new = np.dot(w, x_new)
            delta = r + gamma*q_new - q
            multiplier = alpha*lam*gamma*np.dot(z, x)
            z = gamma*lam*z + (1-multiplier)*x
            w += alpha*(delta + q - q_old)*z - alpha*(q - q_old)*x
            q_old = q_new
            x = x_new
            action = action_new
            eps/=t
            if done:
                bhr_metric[i] = info[2]
        rewards[i] = episode_rewards
    return rewards, bhr_metric