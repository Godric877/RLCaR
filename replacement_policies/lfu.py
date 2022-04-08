from collections import defaultdict, OrderedDict, Counter
from replacement_policies.policy_base import PolicyBase

class LFUCache(PolicyBase):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.object_to_count = {}
        self.count_to_object = defaultdict(OrderedDict)
        self.min_count = None
        self.history = []
        self.history_dict = Counter()

    def update(self, key: int, val: int):
        if key in self.object_to_count:
            count = self.object_to_count[key]
            self.object_to_count[key] += 1
            size =  self.count_to_object[count][key]
            del self.count_to_object[count][key]
            self.count_to_object[count + 1][key] = size
            if not self.count_to_object[self.min_count]:
                self.min_count += 1
        is_present = (key in self.history_dict)
        return is_present

    def get_remove_candidate(self):
        return next(iter(self.count_to_object[self.min_count]))

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
        key ,val = self.count_to_object[self.min_count].popitem(last=False)
        del self.object_to_count[key]
        return key ,val

    def put(self, key: int, value : int) -> None:
        if len(self.object_to_count) >= self.capacity:
            self.remove()
        self.min_count = 1
        self.object_to_count[key] = 1
        self.count_to_object[1][key] = value

    def remove_key(self, key):
        self.update_history()
        if key in self.object_to_count:
            count = self.object_to_count[key]
            del self.object_to_count[key]
            del self.count_to_object[count][key]
            if not self.count_to_object[self.min_count]:
                self.min_count += 1

    def reset(self):
        self.object_to_count = {}
        self.count_to_object = defaultdict(OrderedDict)
        self.min_count = None
        self.history = []
        self.history_dict = Counter()