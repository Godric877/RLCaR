from collections import OrderedDict, Counter
from replacement_policies.policy_base import PolicyBase

class LRUCache(PolicyBase):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache=OrderedDict()
        self.history = []
        self.history_dict = Counter()

    def update(self, key: int, val : int):
        self.cache.move_to_end(key)
        is_present = (key in self.history_dict)
        return is_present

    def get_remove_candidate(self):
        return next(iter(self.cache))

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
        return self.cache.popitem(last=False)

    def put(self, key: int, value : int) -> None:
        if len(self.cache) >= self.capacity:
            self.remove()
        self.cache[key]=value
        self.cache.move_to_end(key)

    def remove_key(self, key):
        self.update_history()
        if key in self.cache:
            del self.cache[key]

    def reset(self):
        self.cache = OrderedDict()
        self.history = []
        self.history_dict = Counter()