import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict, OrderedDict
import heapq
from gym import spaces
from replacement_agent import ReplacementAgent

from trace_loader import load_traces, get_stats

# from park import core, spaces, logger
# from park.param import config
# from park.utils import seeding
# from park.envs.cache.trace_loader import load_traces

accept = 1
reject = 0

cache_unseen_default = 500
cache_size_default = 20
cache_trace_default = "test"


class TraceSrc(object):
    '''
    Tracesrc is the Trace Loader

    @param trace: The file name of the trace file
    @param cache_size: The fixed size of the whole cache
    @param load_trace: The list of trace data. The item could be gotten by using load_trace.iloc[self.idx]
    @param n_request: length of the trace
    @param min_values, max values: Used for the restricted of the value space
    @param req: The obj_time of the object
    '''

    def __init__(self, trace, cache_size):
        self.trace = trace
        self.cache_size = cache_size
        self.load_trace = load_traces(self.trace, self.cache_size, 0)
        self.means, self.stddevs = get_stats(self.load_trace)
        self.n_request = len(self.load_trace)
        self.cache_size = cache_size
        self.min_values = np.asarray([1, 0, 0])
        self.max_values = np.asarray([self.cache_size, self.cache_size, max(self.load_trace[0])])
        self.req = 0

    def reset(self, random):
        if self.trace == 'test' or self.trace.startswith('zipf'):
            self.load_trace = load_traces(self.trace, self.cache_size, random)
            self.means, self.stddevs = get_stats(self.load_trace)
        self.n_request = len(self.load_trace)
        self.min_values = np.asarray([1, 0, 0])
        self.max_values = np.asarray([self.cache_size, self.cache_size, max(self.load_trace[0])])
        self.req = 0

    def step(self):
        #  Obs is: (obj_time, obj_id, obj_size)
        #  print("req id in trace step:", self.req)
        obs = self.load_trace.iloc[self.req].values
        self.req += 1
        done = self.req >= self.n_request
        return obs, done

    def next(self):
        obs = self.load_trace.iloc[self.req].values
        done = (self.req + 1) >= self.n_request
        return obs, done

    def get_trace_stats(self):
        return self.means, self.stddevs

class CacheSim(object):
    def __init__(self, cache_size, policy, action_space, state_space, replacement_policies, trace_means, trace_stddevs, episode_index=0):
        # invariant
        '''
        This is the simulater for the cache.
        @param cache_size
        @param policy: Not implement yet. Maybe we should instead put this part in the action
        @param action_space: The restriction for action_space. For the cache admission agent, it is [0, 1]: 0 is for reject and 1 is for admit
        @param req: It is the idx for the object requiration
        @param non_cache: It is the list for those requiration that aren't cached, have obj_id and last req time
        @param cache: It is the list for those requiration that are cached, have obj id and last req time
        @param count_ohr: ohr is (sigma hit) / req
        @param count_bhr: ohr is (sigma object_size * hit) / sigma object_size
        @param size_all: size_all is sigma object_size
        '''

        self.cache_size = cache_size
        self.policy = policy
        self.action_space = action_space
        self.observation_space = state_space
        self.req = 0
        self.non_cache = defaultdict(list)
        self.cache = defaultdict(list)  # requested items with caching
        self.cache_pq = []
        # self.lru_cache = LRUCache(self.cache_size)
        self.agent = ReplacementAgent(capacity=self.cache_size, policies=replacement_policies,episode_index=episode_index)
        self.cache_remain = self.cache_size
        self.count_ohr = 0
        self.count_bhr = 0
        self.size_all = 0
        self.object_frequency = Counter()
        self.object_average_interarrival = Counter()
        self.trace_means = trace_means
        self.trace_stddevs = trace_stddevs

    def reset(self, trace_means, trace_stddevs, episode_index):
        self.req = 0
        self.non_cache = defaultdict(list)
        self.cache = defaultdict(list)
        self.cache_pq = []
        self.cache_remain = self.cache_size
        self.count_ohr = 0
        self.count_bhr = 0
        self.size_all = 0
        self.agent.reset(index=episode_index)
        self.object_frequency = Counter()
        self.object_average_interarrival = Counter()
        self.trace_means = trace_means
        self.trace_stddevs = trace_stddevs

    def step(self, action, obj):
        #print("object_freq in step(): {}".format(self.object_frequency))
        req = self.req
        # print(self.req)
        cache_size_online_remain = self.cache_remain
        discard_obj_if_admit = []
        obj_time, obj_id, obj_size = obj[0], obj[1], obj[2]
        self.object_frequency[obj_id] += 1


        # create the current state for cache simulator
        cost = 0

        # simulation
        # if the object size is larger than cache size
        if obj_size >= self.cache_size:
            # record the request
            cost += obj_size
            hit = 0
            try:
                self.non_cache[obj_id][1] = req
            except IndexError:
                self.non_cache[obj_id] = [obj_size, req]

        else:
            #  Search the object in the cache
            #  If hit
            try:
                self.cache[obj_id][1] = req
                self.count_bhr += obj_size
                self.count_ohr += 1
                hit = 1
                cost += obj_size
                self.agent.update(obj_id, obj_size)

            #  If not hit
            except IndexError:
                # accept request
                if action == 1:
                    # find the object in the cache, no cost, OHR and BHR ++
                    # can't find the object in the cache, add the object into cache after replacement, cost ++
                    while cache_size_online_remain < obj_size:
                        # rm_id = self.cache_pq[0][1]
                        # cache_size_online_remain += self.cache_pq[0][0]
                        # cost += self.cache_pq[0][0]
                        # discard_obj_if_admit.append(rm_id)
                        # heapq.heappop(self.cache_pq)
                        # del self.cache[rm_id]
                        rm_id, size = self.agent.remove()
                        #print("rm_id = ",rm_id, " size = ", size)
                        cache_size_online_remain += size
                        cost += size
                        discard_obj_if_admit.append(rm_id)
                        del self.cache[rm_id]


                        # add into cache
                    self.cache[obj_id] = [obj_size, req]
                    # heapq.heappush(self.cache_pq, (obj_size, obj_id))
                    self.agent.put(obj_id, obj_size)
                    cache_size_online_remain -= obj_size

                    # cost value is based on size, can be changed
                    cost += obj_size
                    hit = 0

                # reject request
                else:
                    hit = 0
                    # record the request to non_cache
                    try:
                        self.non_cache[obj_id][1] = req
                    except IndexError:
                        self.non_cache[obj_id] = [obj_size, req]

        self.size_all += obj_size
        bhr = float(self.count_bhr / self.size_all)
        ohr = float(self.count_ohr / (req + 1))
        # print("debug:", bhr, ohr)
        reward = hit * cost

        if self.object_frequency[obj_id] != 1:
            new_count = self.object_frequency[obj_id] - 1
            cur_avg = self.object_average_interarrival[obj_id]
            try:
                last_interarrival = self.req - self.cache[obj_id][1]
            except IndexError:
                    last_interarrival = self.req - self.non_cache[obj_id][1]
            new_avg = cur_avg + (last_interarrival - cur_avg)/new_count
            self.object_average_interarrival[obj_id] = new_avg

        self.req += 1
        self.cache_remain = cache_size_online_remain

        info = [self.count_bhr, self.size_all, float(float(self.count_bhr) / float(self.size_all))]
        return reward, info

    def next_hit(self, obj):
        try:
            obj_id = obj[1]
            self.cache[obj_id][1] = self.cache[obj_id][1]
            return True

        except IndexError:
            return False

    def get_normalized_state(self, state):
        normalized_state = []
        for index, s in enumerate(state):
            normalized_state.append( (s-self.trace_means[index])/self.trace_stddevs[index])
        normalized_state[1] /= self.cache_size
        return normalized_state

    def get_state(self, obj=[0, 0, 0, 0]):
        '''
        Return the state of the object,  [obj_size, cache_size_online_remain, recency (steps since object was last visited) = req - last visited time]
        If an object has never been seen before, assigned a constant for the recency feature.
        For more information, see Learning Caching policy_approximations with Subsampling:
            http://mlforsystems.org/assets/papers/neurips2019/learning_wang_2019.pdf
        '''
        obj_time, obj_id, obj_size = obj[0], obj[1], obj[2]
        try:
            req = self.req - self.cache[obj_id][1]
        except IndexError:
            try:
                req = self.req - self.non_cache[obj_id][1]
            except IndexError:
                # Unseen objects (not in non_cache or cache) are assigned this recency constant
                req = cache_unseen_default

        #print("object_freq in get_state: {}".format(self.object_frequency))
        # sorted_frequency = dict(sorted(self.object_frequency.items(), key=lambda item: item[1]))
        # rank  = -1
        # if obj_id in sorted_frequency:
        #     rank = list(sorted_frequency.keys()).index(obj_id)
        # cache_min_freq = math.inf
        # for object in self.cache:
        #     freq = self.object_frequency[object]
        #     cache_min_freq = min(cache_min_freq, freq)
        #print("obj_id = {}, rank = {}".format(obj_id, rank))
        state = [obj_size, self.cache_remain, req, self.object_frequency[obj_id],
                 self.object_average_interarrival[obj_id]]

        return self.get_normalized_state(state)


class CacheEnv():
    """
    Cache description.

    * STATE *
        The state is represented as a vector:
        [request object size,
         cache remaining size,
         time of last request to the same object]

    * ACTIONS *
    TODO: should be fixed here, there should be both
        Whether the cache accept the incoming request, represented as an
        integer in [0, 1].

    * REWARD * (BHR)
        Cost of previous step (object size) * hit

    * REFERENCE *
    """

    def __init__(self, replacement_policies, cache_size=cache_size_default,
                 trace=cache_trace_default, seed=42):
        self.seed(seed)
        self.cache_size = cache_size

        # load trace, attach initial online feature values
        self.src = TraceSrc(trace=trace, cache_size=self.cache_size)

        # set up the state and action space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.src.min_values, \
                                            self.src.max_values, \
                                            dtype=np.float32)

        # cache simulator
        trace_means, trace_stddevs = self.src.get_trace_stats()
        self.sim = CacheSim(cache_size=self.cache_size, \
                            policy='lru', \
                            action_space=self.action_space, \
                            state_space=self.observation_space,
                            replacement_policies=replacement_policies,
                            trace_means=trace_means,
                            trace_stddevs=trace_stddevs,
                            episode_index=0)

        # reset environment (generate new jobs)
        self.reset(1, 2)

    def reset(self, trace_index, low=0, high=1000):
        #new_trace = np.random.randint(low, high)
        self.src.reset(trace_index)
        trace_means, trace_stddevs = self.src.get_trace_stats()
        self.sim.reset(trace_means, trace_stddevs, episode_index=trace_index)
        if cache_trace_default == 'test':
            print("New Env Start", trace_index)
        elif cache_trace_default == 'real':
            print("New Env Start Real")
        return self.sim.get_state()

    def seed(self, seed):
        self.np_random = np.random.seed(seed)

    def step(self, action):
        # 0 <= action < num_servers
        global accept
        assert self.action_space.contains(action)
        state, done = self.src.step()
        reward, info = self.sim.step(action, state)
        obj, done = self.src.next()
        while self.sim.next_hit(obj):
            state, done = self.src.step()
            hit_reward, info = self.sim.step(accept, state)
            reward += hit_reward
            if done is True:
                break
            obj, done = self.src.next()

        obs = self.sim.get_state(obj)
        #info = {}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass
