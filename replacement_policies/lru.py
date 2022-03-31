from collections import OrderedDict
from replacement_policies.policy_base import PolicyBase

class LRUCache(PolicyBase):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache=OrderedDict()

    def update(self, key: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)

    def remove(self):
        return self.cache.popitem(last=False)

    def put(self, key: int, value : int) -> None:
        self.cache[key]=value
        self.cache.move_to_end(key)

    def remove_key(self, key):
        if key in self.cache:
            del self.cache[key]