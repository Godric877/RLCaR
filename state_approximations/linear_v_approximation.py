import numpy as np
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, dims):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Linear(dims, 1)

    def forward(self, x):
        return self.model(x)

class LinearStateApproximation():
        def create_model(self):
            self.model = NeuralNetwork(self.state_dims)
            self.loss_fn = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.alpha)

        def __init__(self,
                     state_dims, alpha):
            """
            state_dims: the number of dimensions of state space
            """
            self.state_dims = state_dims
            self.alpha = alpha
            self.create_model()

        def get_input(self, s):
            return torch.tensor(s)

        def __call__(self, s):
            self.model.eval()
            input = self.get_input(s)
            pred = self.model(input.float())
            return pred.detach().numpy()[0]

        def update(self, update_vector):
            with torch.no_grad():
                update_vector = torch.tensor(update_vector)
                self.model.model.weight += update_vector

