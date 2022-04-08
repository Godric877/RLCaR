from collections import Counter
from replacement_policies.policy_base import PolicyBase

class FifoCache(PolicyBase):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache_stack = []
        self.value_dict = {}
        self.history = []
        self.history_dict = Counter()

    def update(self, key: int, val: int):
        return (key in self.history_dict)

    def get_remove_candidate(self):
        return self.cache_stack[-1]

    def update_history(self):
        candidate = self.get_remove_candidate()
        if (len(self.history) >= self.capacity):
            remove_item = self.history.pop(0)
            if(self.history_dict[remove_item] == 1):
                del self.history_dict[remove_item]
            else:
                self.history_dict[remove_item] -= 1
        self.history.append(candidate)
        self.history_dict[candidate] += 1

    def remove(self):
        self.update_history()
        key = self.cache_stack.pop()
        val = self.value_dict[key]
        del self.value_dict[key]
        return key ,val

    def put(self, key: int, value : int) -> None:
        if len(self.cache_stack) >= self.capacity:
            self.remove()
        self.cache_stack.append(key)
        self.value_dict[key] = value

    def remove_key(self, key):
        self.update_history()
        if key in self.cache_stack:
            self.cache_stack.remove(key)
            del self.value_dict[key]

    def reset(self):
        self.cache_stack = []
        self.value_dict = {}
        self.history = []
        self.history_dict = Counter()