import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from cache import CacheEnv

from algorithms.deterministic import always_evict, random_eviction
from algorithms.semi_gradient_n_step_sarsa_algorithm import semi_gradient_n_step_sarsa, semi_gradient_n_step_sarsa_tc
from algorithms.actor_critic_eligibility_trace_algorithm_linear import actor_critic_eligibility_trace_linear
from algorithms.actor_critic_eligibility_trace_algorithm_tc import actor_critic_eligibility_trace_tc
from algorithms.optimal_algorithm import optimal_admission
from algorithms.reinforce_algorithm import reinforce
from algorithms.true_online_sarsa_lambda import TrueOnlineSarsaLambda
from state_approximations.linear_v_approximation import LinearStateApproximation
from state_approximations.one_d_tc import StateOneDTileCoding
from policy_approximations.linear_policy_approximation import LinearPolicyApproximation
from state_action_approximations.one_d_tc import StateActionOneDTileCoding
from state_action_approximations.linear_q_approximation import LinearStateActionApproximation
from state_action_approximations.nn_q_approximation import NeuralNetworkStateActionApproximation

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

def run_n_step_sarsa_linear(env, train, test, n=1):
    print("Running ", n, "-step Sarsa")
    # Semi-gradient n-step Sarsa with linear function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    if n == 1:
        alpha = 1e-2
    else:
        alpha = 1e-3
    L = LinearStateActionApproximation(5, 2, alpha)
    semi_gradient_n_step_sarsa(env, gamma, alpha, L, epsilon, train, actions, n)
    rewards, bhr = semi_gradient_n_step_sarsa(env, gamma, alpha, L, epsilon, test, actions, n)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/semi_gradient_n_step_sarsa_with_linear_approx.png")

def run_n_step_sarsa_nn(env, train, test, n=1):
    print("Running ", n, "-step Sarsa with nn approximation")
    # Semi-gradient n-step Sarsa with neural network function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha = 1e-2
    L = NeuralNetworkStateActionApproximation(5, 2, alpha)
    semi_gradient_n_step_sarsa(env, gamma, alpha, L, epsilon, train, actions, n)
    rewards, bhr = semi_gradient_n_step_sarsa(env, gamma, alpha, L, epsilon, test, actions, n)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/semi_gradient_n_step_sarsa_with_nn.png")

def run_actor_critic_tc(env, train, test):
    print("Running actor critic with eligibility traces tc")
    # Actor critic with 1-D tile coding
    gamma = 1
    alpha_theta = 1e-3
    alpha_w = 1e-3
    lambda_theta = 0.8
    lamdba_w = 0.8
    state_low = np.array([1, 0, 0, 1, 0])
    state_high = np.array([1, 20, 500, 1200, 500])
    tile_width = np.array([1, 1, 10, 50, 10])
    V = StateOneDTileCoding(
        state_low,
        state_high,
        num_tilings=1,
        tile_width=tile_width
    )
    pi = LinearPolicyApproximation(5, 2, alpha_theta)
    actor_critic_eligibility_trace_tc(env, gamma,
                                      alpha_theta, alpha_w,
                                      lambda_theta, lamdba_w,
                                      V, pi,
                                      train)
    rewards, bhr = actor_critic_eligibility_trace_tc(env, gamma,
                                                     alpha_theta, alpha_w,
                                                     lambda_theta, lamdba_w,
                                                     V, pi,
                                                     test)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/actor_critic_eligibility_trace.png")

def run_actor_critic_linear(env, train, test):
    print("Running actor critic with eligibility traces linear")
    # Actor critic with neural network and eligibility traces
    gamma = 1
    alpha_theta = 1e-3
    alpha_w = 1e-3
    lambda_theta = 0.8
    lamdba_w = 0.8
    V = LinearStateApproximation(5, alpha_w)
    pi = LinearPolicyApproximation(5, 2, alpha_theta)
    actor_critic_eligibility_trace_linear(env, gamma,
                                          alpha_theta, alpha_w,
                                          lambda_theta, lamdba_w,
                                          V, pi,
                                          train)
    rewards, bhr = actor_critic_eligibility_trace_linear(env, gamma,
                                                         alpha_theta, alpha_w,
                                                         lambda_theta, lamdba_w,
                                                         V, pi,
                                                         test)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/actor_critic_eligibility_trace_linear.png")

def run_reinforce(env, train, test):
    print("Running Reinforce")
    # Reinforce with linear function approximation
    epsilon = 0.1
    actions = [0, 1]
    gamma = 1
    alpha_theta = 1e-3
    alpha_w = 1e-3
    n = 1
    L = LinearStateApproximation(5, alpha_w)
    pi = LinearPolicyApproximation(5, 2, alpha_theta)
    reinforce(env, gamma, train,pi, L)
    rewards, bhr = reinforce(env, gamma, test,pi, L)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/reinforce.png")

def run_sarsa_lambda(env, train, test, lam):
    print("Running True Online Sarsa Lambda")
    # True Online Sarsa Lambda with One Dimensional Tile Coding
    epsilon = 0.1
    gamma = 1
    alpha = 1e-2
    actions = [0, 1]
    state_low = np.array([1, 0, 0, 1, 0])
    state_high = np.array([1, 20, 500, 1200, 500])
    tile_width = np.array([1, 1, 10, 50, 10])
    Q = StateActionOneDTileCoding(
        state_low,
        state_high,
        num_tilings=2,
        tile_width=tile_width,
        num_actions=len(actions)
    )
    TrueOnlineSarsaLambda(env, epsilon, gamma, lam, alpha, Q, train)
    rewards, bhr = TrueOnlineSarsaLambda(env, epsilon, gamma, lam, alpha, Q, test)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/true_online_sarsa_lambda.png")

def run_always_evict(env, train, test):
    print("Running Always Evict")
    rewards, bhr = always_evict(env, test)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/lru.png")

def run_random_eviction(env, train, test):
    p = [0.5,0.5]
    print("Running Random Policy ")
    rewards, bhr = random_eviction(env, test, p)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/random_eviction_lru.png")

def run_optimal(test, cache_size):
    print("Running Optimal Policy")
    rewards, bhr = optimal_admission(episodes=test, cache_size=cache_size)
    print("======================")
    return get_metrics(test, 0, rewards, bhr, "experiments/graphs/optimal_algorithm.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL CaR')
    parser.add_argument(
        '-ne',
        '--num_episodes',
        help='Number of episodes',
        type=int,
        default=70
    )
    parser.add_argument(
        '-nr',
        '--num_repetitions',
        help='Number of repetitions',
        type=int,
        default=10
    )
    parser.add_argument(
        '-fa',
        '--function_approximation',
        help='function approximation to use',
        default='tc'
    )

    parser.add_argument(
        '-n_steps',
        '--n_steps',
        help='number of steps in sarsa',
        type=int,
        default=2
    )

    parser.add_argument(
        '-lam',
        '--lam',
        help='lambda in sarsa',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '-rl',
        '--rl_algo',
        help='rl algo to use',
        default='actor_critic'
    )

    parser.add_argument(
        '-policy',
        '--policy',
        help='cache replacement policy space separated',
        default="LRU"
    )

    parser.add_argument(
        '-ts',
        '--test_size',
        help='test size',
        type=int,
        default="20"
    )

    parser.add_argument(
        '-cs',
        '--cache_size',
        help='cache size',
        type=int,
        default="20"
    )
    seeds = [10, 20, 30, 40 ,50, 60, 70, 80, 90, 100]

    args = parser.parse_args()
    print("Arguments ", args)
    num_repetitions = args.num_repetitions
    cache_size = args.cache_size

    num_episodes = args.num_episodes
    episodes = np.arange(num_episodes)
    function_name = 'run_' + args.rl_algo
    if args.function_approximation is not None:
        function_name += '_' + args.function_approximation

    print("Using Function:", function_name)

    function_dict = {"run_reinforce" : run_reinforce,
                     "run_actor_critic_tc" : run_actor_critic_tc,
                     "run_actor_critic_linear" : run_actor_critic_linear,
                     "run_random_eviction" : run_random_eviction,
                     "run_always_evict": run_always_evict,
                     "run_n_step_sarsa_nn" : run_n_step_sarsa_nn,
                     "run_n_step_sarsa_linear" : run_n_step_sarsa_linear,
                     "run_optimal" : run_optimal,
                     "run_sarsa_lambda" : run_sarsa_lambda}

    logging.basicConfig(level=logging.INFO,
                       datefmt='%Y-%m-%d %H:%M:%S', handlers=[
            logging.FileHandler('logs/{}'.format(function_name)),
            logging.StreamHandler()
        ])

    policies = args.policy.split(" ")
    env = CacheEnv(policies, cache_size=cache_size)
    bhr_metrics = []
    for r in range(num_repetitions):
        train, test = train_test_split(episodes, test_size=args.test_size, random_state=seeds[r])
        if function_name.startswith("run_n"):
            n =args.n_steps
            bhr_metrics.append(function_dict[function_name](env, train, test, n))
        elif function_name.startswith("run_optimal"):
            bhr_metrics.append(function_dict[function_name](test, cache_size))
        elif function_name.startswith("run_sarsa_lambda"):
            bhr_metrics.append(function_dict[function_name](env, train, test, args.lam))
        else:
            bhr_metrics.append(function_dict[function_name](env, train, test))
    print("bhr_metrics", bhr_metrics)
    mean_bhr = np.mean(np.array(bhr_metrics))
    print("mean_bhr = ", mean_bhr)
    log_string = "args  = {}, bhr_metrics = {}, mean_bhr = {}".format(args, ', '.join(str(e) for e in bhr_metrics), str(mean_bhr))
    logging.info(log_string)

