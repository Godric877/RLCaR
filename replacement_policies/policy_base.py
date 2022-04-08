class PolicyBase:

    def __init__(self, capacity: int):
        pass

    def update(self, key: int, val: int) -> bool:
        pass

    def get_remove_candidate(self):
        pass

    def update_history(self):
        pass

    def remove(self):
        pass

    def put(self, key: int, value : int) -> None:
        pass

    def remove_key(self, key):
        pass

    def reset(self):
        pass