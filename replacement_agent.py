import math

import numpy as np

from replacement_policies.lru import LRUCache
from replacement_policies.lfu import LFUCache

class ReplacementAgent:
    def __init__(self, capacity, policies):
        self.capacity = capacity
        self.policies = policies
        self.experts = []
        self.num_experts = len(policies)
        self.experts.append(LRUCache(capacity))
        self.current_expert = 0
        self.epsilon = 0.1
        if "LFU" in policies:
            self.experts.append(LFUCache(capacity))
        self.running_reward = 0

        self.weights = np.ones(shape=self.num_experts)

    def update(self, key: int, obj_size) -> None:
        for expert in self.experts:
            expert.update(key)
        self.running_reward += obj_size

    def remove(self):
        self.weights[self.current_expert] *= math.pow(1  + self.epsilon,self.running_reward)
        current_expert = int(np.random.choice(np.arange(self.num_experts), 1, self.weights/np.sum(self.weights)))
        key ,val = self.experts[current_expert].remove()
        for i in range(self.num_experts):
            if i != current_expert:
                self.experts[i].remove_key(key)
        self.running_reward = 0
        return key, val

    def put(self, key: int, value : int) -> None:
        for expert in self.experts:
            expert.put(key, value)