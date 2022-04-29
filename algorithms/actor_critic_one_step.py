# Actor Critic with Eligibility Traces using Linear Approximation for State and Policy

import numpy as np

from state_approximations.linear_v_approximation import LinearStateApproximation
from state_approximations.nn_v_approximation import NNStateApproximation
from policy_approximations.linear_policy_approximation import LinearPolicyApproximation

def actor_critic_one_step_nn(env, gamma, alpha_theta, alpha_w,
                                   V:NNStateApproximation, pi:LinearPolicyApproximation,
                                   episodes):
    bhr_metric = {}
    rewards = {}
    for i in episodes:
        s_current = env.reset(i)
        I = 1
        done = False
        episode_rewards = []
        while not done:
            action = pi(s_current)
            s_next, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            delta = reward + gamma*V(s_next)- V(s_current)
            V.manual_update(V.return_gradient(s_current)*delta*alpha_w)
            pi.manual_update(pi.return_gradient(s_current, action)*alpha_theta*I*delta)
            I = gamma*I
            s_current = s_next
            if done:
                bhr_metric[i] = info[2]
        rewards[i] = episode_rewards
    return rewards, bhr_metric

def actor_critic_one_step(env, gamma, alpha_theta, alpha_w,
                                   V:LinearStateApproximation, pi:LinearPolicyApproximation,
                                   episodes):
    bhr_metric = {}
    rewards = {}
    for i in episodes:
        s_current = env.reset(i)
        I = 1
        done = False
        episode_rewards = []
        while not done:
            action = pi(s_current)
            s_next, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            delta = reward + gamma*V(s_next)- V(s_current)
            V.manual_update(V.return_gradient(s_current)*delta*alpha_w)
            pi.manual_update(pi.return_gradient(s_current, action)*alpha_theta*I*delta)
            I = gamma*I
            s_current = s_next
            if done:
                bhr_metric[i] = info[2]
        rewards[i] = episode_rewards
    return rewards, bhr_metric



