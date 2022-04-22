import os

import numpy as np
import pandas as pd
import statistics

from collections import Counter


def load_traces(trace : str, cache_size, rnd):
    if trace == 'test':
        trace_folder = os.curdir + '/trace/'
        print(trace_folder)

        print('Load #%i trace for cache size of %i' % (rnd, cache_size))

        # load time, request id, request size
        df = pd.read_csv(trace_folder + 'test_trace/test_' + str(rnd)  + '.tr', sep=' ', header=None)
        # remaining cache size, object last access time
        df[3], df[4] = cache_size, 0
        df[2] = 1
    else:
        trace_folder = os.curdir + '/trace/'
        print(trace_folder)

        print('Load #%i trace for cache size of %i' % (rnd, cache_size))

        # load time, request id, request size
        df = pd.read_csv(trace_folder + trace + '/trace_' + str(rnd)  + '.tr', sep=' ', header=None)
        # remaining cache size, object last access time
        df[3], df[4] = cache_size, 0
        df[2] = 1

    # elif trace == 'real':
    #    df = []
    # else:
    #     # load user's trace
    #     df = pd.read_csv(trace, sep=' ', header=None)
    #     df[3], df[4] = cache_size, 0

    return df

def get_stats(df):
    cache_unseen_default = 500

    obj_freq = Counter()
    obj_interarrival_time = {}
    all_interarrival_times = []
    last_access_time = []
    for index, row in df.iterrows():
        obj_freq[row[1]] += 1
        if(row[1] not in obj_interarrival_time):
            obj_interarrival_time[row[1]] = index
            last_access_time.append(cache_unseen_default)
        else:
            all_interarrival_times.append(index - obj_interarrival_time[row[1]])
            last_access_time.append(index - obj_interarrival_time[row[1]])
            obj_interarrival_time[row[1]] = index

    # stats for object frequency
    obj_freq_mean = statistics.mean(obj_freq.values())
    obj_freq_stdev = statistics.stdev(obj_freq.values())

    # stats for object size
    obj_size_mean = statistics.mean(df[2])
    obj_size_stdev = statistics.stdev(df[2])

    # stats for interarrival times
    obj_interarrival_time_mean = statistics.mean(all_interarrival_times)
    obj_interarrival_time_stdev = statistics.stdev(all_interarrival_times)

    # stats for last access time
    last_access_time_mean = statistics.mean(last_access_time)
    last_access_time_stdev = statistics.stdev(last_access_time)

    #stats for rank
    ranks = np.arange(len(obj_freq))
    rank_mean = statistics.mean(ranks)
    rank_stdev = statistics.stdev(ranks)

    means = [obj_size_mean, 0, last_access_time_mean, obj_freq_mean, obj_interarrival_time_mean, rank_mean]
    stddevs = [obj_size_stdev, 1, last_access_time_stdev, obj_freq_stdev, obj_interarrival_time_stdev, rank_stdev]

    for index, stddev in enumerate(stddevs):
        if(stddev == 0):
            stddevs[index] = 1

    return means, stddevs




