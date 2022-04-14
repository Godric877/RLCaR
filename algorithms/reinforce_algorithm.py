import math

from state_approximations.linear_v_approximation import Baseline
from policy_approximations.linear_policy_approximation import LinearPolicyApproximation

def reinforce(
    env, #open-ai environment
    gamma:float,
    episodes,
    pi:LinearPolicyApproximation,
    V:Baseline):
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    bhr_metric = {}
    return_rewards = {}
    for i in episodes:
        state = env.reset(i)
        done = False
        rewards = [0]
        states = [state]
        actions = []
        episode_rewards = []
        while not done:
            action = pi(state)
            state, r, done, info = env.step(action)
            rewards.append(r)
            episode_rewards.append(r)
            actions.append(action)
            if not done:
                states.append(state)
            else:
                bhr_metric[i] = info[2]
        return_rewards[i] = episode_rewards
        G = 0
        for t in range(len(states)):
            G += math.pow(gamma, t)*rewards[t + 1]
        delta = G - V(states[0])
        V.update(states[0], G)
        pi.update(states[0], actions[0], 1, delta)
        i = 1
        gamma_t = 1
        while i < len(states):
            G = (G - rewards[i])/gamma
            gamma_t = gamma_t*gamma
            delta = G - V(states[i])
            V.update(states[i], G)
            pi.update(states[i], actions[i], gamma_t, delta)
            i = i + 1

    return return_rewards, bhr_metric