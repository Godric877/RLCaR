from cache import CacheEnv
import matplotlib.pyplot as plt
import numpy as np
from algorithms.semi_gradient_sarsa_algorithm import semi_gradient_sarsa
from algorithms.actor_critic_eligibility_trace_algorithm import actor_critic_eligibility_trace
from algorithms.deterministic import always_evict
from algorithms.true_online_sarsa_lambda import TrueOnlineSarsaLambda
from algorithms.semi_gradient_n_step_sarsa_algorithm import semi_gradient_n_step_sarsa
from state_action_approximations.linear_q_approximation import LinearStateActionApproximation
from state_approximations.linear_v_approximation import LinearStateApproximation
from policy_approximations.linear_policy_approximation import LinearPiApproximation
from state_action_approximations.tile_coding_state_action import StateActionFeatureVectorWithTile
from sklearn.model_selection import train_test_split


def plot_reward(rewards, filename):
    plt.plot(np.arange(0, len(rewards)), np.cumsum(rewards))
    plt.xlabel("time")
    plt.ylabel("cumulative reward")
    plt.savefig(filename)
    plt.close()
    #plt.show()

def get_metrics(test, episode, rewards, bhr, filename):
    avg_test_bhr = 0.0
    for index in test:
        avg_test_bhr += bhr[index]
    avg_test_bhr /= len(test)
    print("Average BHR on test data : ", avg_test_bhr)
    plot_reward(rewards[test[episode]], filename)

def semi_gradient_sarsa_with_linear_approx(env, train, test):
    # Semi-gradient Sarsa with linear function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha = 0.01
    L = LinearStateActionApproximation(5, 2, alpha)
    semi_gradient_sarsa(env, gamma, alpha, L, epsilon, train, actions)
    rewards, bhr = semi_gradient_sarsa(env, gamma, alpha, L, epsilon, test, actions)
    get_metrics(test, 0, rewards, bhr, "experiments/graphs/semi_gradient_sarsa.png")

def semi_gradient_n_step_sarsa_with_linear_approx(env, train, test):
    # Semi-gradient Sarsa with linear function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha = 0.01
    n = 1
    L = LinearStateActionApproximation(5, 2, alpha)
    semi_gradient_n_step_sarsa(env, gamma, alpha, L, epsilon, train, actions, n)
    rewards, bhr = semi_gradient_n_step_sarsa(env, gamma, alpha, L, epsilon, test, actions, n)
    get_metrics(test, 0, rewards, bhr, "experiments/graphs/semi_gradient_n_step_sarsa.png")

def true_online_sarsa_lambda(env, train, test):
    gamma = 1
    alpha = 0.01
    lamda = 0.8
    state_low = np.array([1, 0, 0, 1, 0])
    state_high = np.array([1, 20, 500, 1200, 500])
    tile_width = np.array([1, 2, 50, 120, 50])
    X = StateActionFeatureVectorWithTile(
        state_low,
        state_high,
        env.action_space.n,
        num_tilings=2,
        tile_width=tile_width
    )
    TrueOnlineSarsaLambda(env, gamma, lamda, alpha, X, train)
    rewards, bhr = TrueOnlineSarsaLambda(env, gamma, lamda, alpha, X, test)
    get_metrics(test, 0, rewards, bhr, "experiments/graphs/true_online_sarsa_lambda.png")

def actor_critic_with_eligibility_traces_and_linear_approx(env, train, test):
    # Semi-gradient Sarsa with linear function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha_theta = 1e-4
    alpha_w = 1e-4
    lambda_theta = 0.2
    lamdba_w = 0.3
    V = LinearStateApproximation(5, alpha_w)
    pi = LinearPiApproximation(5, 2, alpha_theta)
    actor_critic_eligibility_trace(env, gamma,
                                   alpha_theta, alpha_w,
                                   lambda_theta, lamdba_w,
                                   V, pi,
                                   train)
    rewards, bhr = actor_critic_eligibility_trace(env, gamma,
                                   alpha_theta, alpha_w,
                                   lambda_theta, lamdba_w,
                                   V, pi,
                                   test)
    get_metrics(test, 0, rewards, bhr, "experiments/graphs/actor_critic_eligibility_trace_linear.png")

def run_always_evict(env, train, test):
    always_evict(env, train)
    rewards, bhr = always_evict(env, test)
    get_metrics(test, 0, rewards, bhr, "experiments/graphs/lru.png")

if __name__ == "__main__":
    num_episodes = 20
    episodes = np.arange(num_episodes)
    train, test = train_test_split(episodes, test_size=0.2)

    policies = ["LRU"]
    env = CacheEnv(policies)
    #semi_gradient_sarsa_with_linear_approx(env, train, test)
    semi_gradient_n_step_sarsa_with_linear_approx(env, train, test)

    # for policy in policies:
    #     env = CacheEnv([policy])
    #     print("Policy used: ", policy)
    #     semi_gradient_sarsa_with_linear_approx(env, train, test)
    #
    # env = CacheEnv(["LRU"])
    # run_always_evict(env, train, test)
