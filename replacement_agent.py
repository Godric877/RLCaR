import math

import numpy as np

from replacement_policies.lru import LRUCache
from replacement_policies.lfu import LFUCache
from replacement_policies.fifo import FifoCache

class ReplacementAgent:
    def __init__(self, capacity, policies):
        self.capacity = capacity
        self.policies = policies
        self.experts = []
        self.num_experts = len(policies)
        self.current_expert = 0
        self.hit_reward = 1
        self.miss_reward = -0.5
        self.epsilon = 0.1
        if "LRU" in policies:
            self.experts.append(LRUCache(capacity))
        if "LFU" in policies:
            self.experts.append(LFUCache(capacity))
        if "FIFO" in policies:
            self.experts.append(FifoCache(capacity))
        self.running_reward = 0

        self.weights = np.ones(shape=self.num_experts)
        self.reward = np.zeros(shape=self.num_experts)

    def update(self, key: int, obj_size) -> None:
        for index, expert in enumerate(self.experts):
            is_present_in_history = expert.update(key, obj_size)
            if is_present_in_history:
                self.reward[index] = self.miss_reward
            else:
                self.reward[index] = self.hit_reward
        if(self.num_experts > 1):
            self.weight_update()

    def remove(self):
        current_expert = int(np.random.choice(np.arange(self.num_experts), 1, p=self.weights/np.sum(self.weights)))
        key ,val = self.experts[current_expert].remove()
        for i in range(self.num_experts):
            if i != current_expert:
                self.experts[i].remove_key(key)
        self.running_reward = 0
        return key, val

    def put(self, key: int, value : int) -> None:
        for expert in self.experts:
            expert.put(key, value)
        self.update(key, value)

    def weight_update(self):
        for index, expert in enumerate(self.experts):
            self.weights[index] *= math.pow(1 + self.epsilon, self.reward[index])

    def reset(self):
        for expert in self.experts:
            expert.reset()
        self.weights = self.weights/np.sum(self.weights)
        self.current_expert = 0