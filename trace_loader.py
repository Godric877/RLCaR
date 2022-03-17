import os
import pandas as pd


def load_traces(trace, cache_size, rnd):
    if trace == 'test':
        trace_folder = os.curdir + '/trace/'
        print(trace_folder)

        print('Load #%i trace for cache size of %i' % (rnd, cache_size))

        # load time, request id, request size
        df = pd.read_csv(trace_folder + 'test_trace/small'  + '.tr', sep=' ', header=None)
        # remaining cache size, object last access time
        df[3], df[4] = cache_size, 0

    elif trace == 'real':
       df = []
    else:
        # load user's trace
        df = pd.read_csv(trace, sep=' ', header=None)
        df[3], df[4] = cache_size, 0

    return df
