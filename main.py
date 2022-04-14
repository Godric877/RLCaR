from cache import CacheEnv
import matplotlib.pyplot as plt
import numpy as np
from algorithms.semi_gradient_sarsa_algorithm import semi_gradient_sarsa
from algorithms.actor_critic_eligibility_trace_algorithm import actor_critic_eligibility_trace
from algorithms.deterministic import always_evict
from algorithms.reinforce_algorithm import reinforce
from algorithms.true_online_sarsa_lambda import TrueOnlineSarsaLambda
from algorithms.semi_gradient_n_step_sarsa_algorithm import semi_gradient_n_step_sarsa
from state_action_approximations.linear_q_approximation import LinearStateActionApproximation
from state_approximations.linear_v_approximation import LinearStateApproximation
from policy_approximations.linear_policy_approximation import LinearPolicyApproximation
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
    return avg_test_bhr
    # plot_reward(rewards[test[episode]], filename)

def semi_gradient_sarsa_with_linear_approx(env, train, test):
    # Semi-gradient Sarsa with linear function approximation
    print("Running one-step Sarsa")
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha = 0.01
    L = LinearStateActionApproximation(5, 2, alpha)
    semi_gradient_sarsa(env, gamma, alpha, L, epsilon, train, actions)
    rewards, bhr = semi_gradient_sarsa(env, gamma, alpha, L, epsilon, test, actions)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/semi_gradient_sarsa.png")

def semi_gradient_n_step_sarsa_with_linear_approx(env, train, test, n=1):
    print("Running ", n, "-step Sarsa")
    # Semi-gradient n-step Sarsa with linear function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha = 0.01
    L = LinearStateActionApproximation(5, 2, alpha)
    semi_gradient_n_step_sarsa(env, gamma, alpha, L, epsilon, train, actions, n)
    rewards, bhr = semi_gradient_n_step_sarsa(env, gamma, alpha, L, epsilon, test, actions, n)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/semi_gradient_n_step_sarsa.png")

def run_reinforce(env, train, test):
    print("Running Reinforce")
    # Reinforce with linear function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha_theta = 3e-4
    alpha_w = 3e-4
    n = 1
    L = LinearStateApproximation(5, alpha_w)
    pi = LinearPolicyApproximation(5, 2, alpha_theta)
    reinforce(env, gamma, train,pi, L)
    rewards, bhr = reinforce(env, gamma, test,pi, L)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/reinforce.png")

def true_online_sarsa_lambda(env, train, test):
    print("Running True online Sarsa Lambda")
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
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/true_online_sarsa_lambda.png")
    print("======================")

def run_always_evict(env, train, test):
    print("Running Always Evict")
    always_evict(env, train)
    rewards, bhr = always_evict(env, test)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/lru.png")

if __name__ == "__main__":
    num_repetitions = 5

    # LRU experiments
    num_episodes = 20
    episodes = np.arange(num_episodes)
    policies = ["LRU"]
    env = CacheEnv(policies)
    bhr_metrics = {"always_evict" : [],
                   "semi_gradient_sarsa_1" : [],
                   "semi_gradient_sarsa_5" : [],
                   "reinforce" : []}
    for r in range(num_repetitions):
        train, test = train_test_split(episodes, test_size=0.2)
        bhr_metrics["always_evict"].append(run_always_evict(env, train, test))
        bhr_metrics["semi_gradient_sarsa_1"].append(semi_gradient_n_step_sarsa_with_linear_approx(env, train, test, n=1))
        bhr_metrics["semi_gradient_sarsa_5"].append(semi_gradient_n_step_sarsa_with_linear_approx(env, train, test, n=5))
        bhr_metrics["reinforce"].append(run_reinforce(env, train, test))
    print(bhr_metrics)
    for rl_algo in bhr_metrics:
        print(rl_algo, ": ", np.mean(np.array(bhr_metrics[rl_algo])))
