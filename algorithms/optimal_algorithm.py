import numpy as np
from collections import defaultdict
from trace_loader import load_traces

def get_next_access(id, current_time, next_access_times):
    if id not in next_access_times:
        return 20000
    else:
        for access_time in next_access_times[id]:
            if(access_time > current_time):
                return access_time
        return 20000

def pre_process(ids):
    next_access_times = defaultdict(list)
    for time_step, id in enumerate(ids):
        next_access_times[id].append(time_step)
    return next_access_times

def optimal_admission(episodes, cache_size, trace='test'):
    rewards = {}
    bhr_metric = {}
    for i in episodes:
        current_cache = {}
        bhr = 0
        ids = list(load_traces(trace, cache_size, i)[1])
        next_access_times = pre_process(ids)
        episode_rewards = []
        hits_since_previous_miss = 0
        for time_step, id in enumerate(ids):
            if id in current_cache:
                bhr += 1
                hits_since_previous_miss += 1
            else:
                episode_rewards.append(hits_since_previous_miss)
                hits_since_previous_miss = 0
                if len(current_cache) < cache_size:
                    current_cache[id] = 1
                else:
                   max_next_access = time_step
                   max_next_access_id = -1
                   for element in current_cache.keys():
                       next_access = get_next_access(element, time_step, next_access_times)
                       if(next_access > max_next_access):
                           max_next_access = next_access
                           max_next_access_id = element
                   if max_next_access > get_next_access(id, time_step, next_access_times):
                        del current_cache[max_next_access_id]
                        current_cache[id] = 1
        bhr_metric[i] = bhr/len(ids)
        rewards[i] = episode_rewards
    return rewards, bhr_metric
