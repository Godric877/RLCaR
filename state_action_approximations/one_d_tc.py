import math
import numpy as np

class StateActionOneDTileCoding():

    def get_state(self, state):
        states = []
        for tile_index in range(self.num_tilings):
            indices = []
            previous_index = 0
            for i in range(self.tiling_dimensions):
                start = self.state_low[i] - tile_index * self.tile_width[i]/self.num_tilings
                diff = state[i] - start
                index = math.floor(diff / self.tile_width[i])
                index = min(index, self.tiles_per_dim[i] - 1)
                index += previous_index
                indices.append(index)
                previous_index += self.tiles_per_dim[i]
            states.append(indices)
        return states

    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array,
                 num_actions:int):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.state_low = state_low
        self.state_high = state_high
        self.tiling_dimensions = len(self.state_low)
        self.num_actions = num_actions
        tiles_per_dim = np.zeros(shape=self.tiling_dimensions, dtype=np.int64)
        for i in range(self.tiling_dimensions):
            tiles_per_dim[i] = math.ceil((self.state_high[i] - self.state_low[i]) /self.tile_width[i]) + 1
        self.tiles_per_dim = tiles_per_dim
        self.num_tiles_per_tiling = np.sum(self.tiles_per_dim)
        self.feature_dims = self.num_tilings*self.num_tiles_per_tiling*self.num_actions
        self.feature_array = np.zeros(shape=(3))
        self.feature_array[0] = self.num_actions
        self.feature_array[1] = self.num_tilings
        self.feature_array[2] = self.num_tiles_per_tiling
        self.feature_array = self.feature_array.astype(int)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.feature_dims

    def __call__(self, s, a, done) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # feature = np.zeros(shape=self.feature_vector_len())
        feature = np.zeros(tuple(self.feature_array))
        if done:
            return feature.flatten()
        active_states = self.get_state(s)
        for i in range(len(active_states)):
            feature[a][i][active_states[i]] = 1
        return feature.flatten()