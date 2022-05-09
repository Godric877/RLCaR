# RLCaR
Reinforcement Learning for Cache admission and Replacement

We propose to apply reinforcement learning on caching systems. The first problem
we consider is to decide whether we want to admit an object in the cache, when
an object request leads to a cache miss. While cache replacement policies have
received significant traction over the years, most systems use simple LRU for
eviction, without explicit admission algorithms. The optimal algorithm for solving
cache admission will require access to future requests, thus making it impractical.
We train an RL agent to give a binary decision of admit/don’t admit for each cache
miss. We show that using our RL agent gives a higher byte hit rate compared
to always admitting on a cache miss or using a random policy to admit an item
in the cache when LRU (Least Recently Used) is used as the cache replacement
policy. The next problem that we consider is the more common problem of cache
replacement, i.e, deciding which object to evict from the cache on a miss. We model
this as an adversarial bandit problem, treating LRU, LFU (Least Frequently Used)
and FIFO (First In First Out) as experts, and solve it using the Hedge algorithm,
assuming full feedback. We show that the algorithm eventually converges to the
best expert. Our experiments are based on a simulated environment, where the
cache traces are generated using a Zip-f distribution, which has been widely used
in simulations.

##Environment Setup
Create a new python environment and install dependencies using `pip install -r requirements.txt`

##Instructions to run:
To run with default arguments : `python3 main.py` 

Arguments : 
+ `-ne NUM_EPISODES, --num_episodes NUM_EPISODES
                        Number of episodes`

+ `-nr NUM_REPETITIONS, --num_repetitions NUM_REPETITIONS
                        Number of repetitions`

+ `-fa FUNCTION_APPROXIMATION, --function_approximation FUNCTION_APPROXIMATION
                        function approximation to use [linear, tc, nn]`

+ `-n_steps N_STEPS, --n_steps N_STEPS
                        number of steps in sarsa`

+ `-lam LAM, --lam LAM   lambda in sarsa`

+ `-rl RL_ALGO, --rl_algo RL_ALGO
                        rl algorithm to use [always_evict, random_eviction, actor_critic, n_step_sarsa, optimal, sarsa_lambda]`

+ `-policy POLICY, --policy POLICY
                        cache replacement policy space separated [LRU, LFU, FIFO]`

+ `-ts TEST_SIZE, --test_size TEST_SIZE
                        test size`

+ `-cs CACHE_SIZE, --cache_size CACHE_SIZE
                        cache size`







