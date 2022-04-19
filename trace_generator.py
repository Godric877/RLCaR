import matplotlib.pyplot as plt
import numpy as np
import math

import pandas as pd

from trace_loader import load_traces
from collections import Counter

def analyse_trace(index):
    trace = load_traces('test', 20, index)

    counts = Counter()
    total_count = 0
    for index, row in trace.iterrows():
        counts[row[1]]+=1
        total_count+=1
    for key in counts:
        counts[key] /= total_count

    print("Total Count : ", total_count)
    print("Total Unique Objects : ", len(counts))

    real_probabilities = np.sort(np.array(list(counts.values())))[::-1]

    predicted_probabilities = np.zeros(len(counts))
    for i in range(1,len(counts)+1):
        predicted_probabilities[i-1] = math.pow(i, -(0.939))
    predicted_probabilities /= np.sum(predicted_probabilities)

    # print("real probabilities : ", real_probabilities)
    # print("predicted probabilities : ", predicted_probabilities)

    # plt.figure()
    # plt.plot(real_probabilities, label='real_probabilities')
    # plt.plot(predicted_probabilities, label='predicted_probabilities')
    # plt.legend()
    # plt.show()

def generate_zipf(alpha, total_requests, unique_requests):
    predicted_probabilities = np.zeros(unique_requests)
    for i in range(1, unique_requests + 1):
        predicted_probabilities[i - 1] = math.pow(i, -alpha)
    predicted_probabilities /= np.sum(predicted_probabilities)

    trace = np.random.choice(np.arange(unique_requests), p=predicted_probabilities, size=total_requests)
    trace_dict = {"timestamp" : np.arange(total_requests),
                  "id" : trace,
                  "obj_size" : np.ones(total_requests)}
    return pd.DataFrame(trace_dict)

def generate_lru_optimal(total_requests, unique_requests):
    trace_subarray = np.zeros(2*unique_requests)
    trace_subarray[:unique_requests] = np.arange(unique_requests)
    trace_subarray[unique_requests:] = np.flip(np.arange(unique_requests))
    trace = np.repeat(trace_subarray, int(total_requests/(2*unique_requests)) + 1)
    trace = trace[:total_requests]
    trace_dict = {"timestamp": np.arange(total_requests),
                  "id": trace,
                  "obj_size": np.ones(total_requests)}
    return pd.DataFrame(trace_dict)

alpha = 0.5
num_request = 100000
unique_requests = 9000

for i in range(1):
    trace_df = generate_zipf(alpha, num_request, unique_requests)
    file_name = "trace/zipf_{}/trace_0.tr".format(alpha, i)
    trace_df.to_csv(file_name, index=False, header=False, sep=" ")

# for i in range(1):
#     trace_df = generate_lru_optimal(num_request, unique_requests)
#     file_name = "trace/lru_optimal/trace_{}.tr".format(i)
#     trace_df.to_csv(file_name, index=False, header=False, sep=" ")
