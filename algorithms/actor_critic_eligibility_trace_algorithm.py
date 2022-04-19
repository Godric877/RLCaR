# Actor Critic with Eligibility Traces using Linear Approximation for State and Policy

import numpy as np

from state_approximations.one_d_tc import StateOneDTileCoding
from policy_approximations.linear_policy_approximation import LinearPolicyApproximation

def actor_critic_eligibility_trace(env, gamma, alpha_theta, alpha_w, lambda_theta, lambda_w,
                                   V:StateOneDTileCoding, pi:LinearPolicyApproximation,
                                   episodes):
    bhr_metric = {}
    rewards = {}
    for i in episodes:
        s_current = env.reset(i)

        #print(s_current)
        print("Feature length {}".format(V.feature_vector_len()))

        V_weights_shape = np.array(V.feature_vector_len())
        pi_weights_shape = np.array(list(pi.model.model[0].weight.shape))

        z_w = np.zeros(V_weights_shape)
        z_theta = np.zeros(pi_weights_shape)

        V_weights = np.zeros(V_weights_shape)

        I = 1
        done = False
        episode_rewards = []
        while not done:
            s_current  = [s_x /1000 for s_x in s_current]
            action = pi(s_current)
            s_next, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            delta = reward + gamma*np.dot(V(s_next, done), V_weights)- np.dot(V(s_current,done), V_weights)
            z_w = gamma*lambda_w*z_w + V(s_current,done)
            z_theta = gamma*lambda_theta*z_theta + I*pi.return_gradient(s_current, action)
            # print("z_w = ", z_w)
            # print("z_theta = ", z_theta)
            # print("state: ", s_current)
            # print("pi.return_gradient ", pi.return_gradient(s_current, action))
            V_weights += alpha_w*delta*z_w
            pi.manual_update(alpha_theta*delta*z_theta)
            I = gamma*I
            s_current = s_next
            if done:
                bhr_metric[i] = info[2]
        rewards[i] = episode_rewards
    return rewards, bhr_metric



