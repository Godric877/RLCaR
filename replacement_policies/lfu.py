from collections import defaultdict, OrderedDict
from replacement_policies.policy_base import PolicyBase

class LFUCache(PolicyBase):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.object_to_count = {}
        self.count_to_object = defaultdict(OrderedDict)
        self.min_count = None

    def update(self, key: int) -> None:
        if key in self.object_to_count:
            count = self.object_to_count[key]
            self.object_to_count[key] += 1
            size =  self.count_to_object[count][key]
            del self.count_to_object[count][key]
            self.count_to_object[count + 1][key] = size
            if not self.count_to_object[self.min_count]:
                self.min_count += 1

    def remove(self):
        key ,val = self.count_to_object[self.min_count].popitem(last=False)
        del self.object_to_count[key]
        return key ,val

    def put(self, key: int, value : int) -> None:
        self.min_count = 1
        self.object_to_count[key] = 1
        self.count_to_object[1][key] = value


    def remove_key(self, key):
        if key in self.object_to_count:
            count = self.object_to_count[key]
            del self.object_to_count[key]
            del self.count_to_object[count][key]
            if not self.count_to_object[self.min_count]:
                self.min_count += 1