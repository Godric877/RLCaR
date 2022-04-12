import numpy as np

from state_action_approximations.state_action_approximation import StateActionApproximation
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, dims):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Linear(dims, 1)

    def forward(self, x):
        return self.model(x)

class LinearStateActionApproximation(StateActionApproximation):
        def create_model(self):
            self.model = NeuralNetwork(self.state_dims * self.num_actions)
            self.loss_fn = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.alpha)

        def __init__(self,
                     state_dims, num_actions, alpha):
            """
            state_dims: the number of dimensions of state space
            """
            # TODO: implement this method
            self.state_dims = state_dims
            self.num_actions = num_actions
            self.alpha = alpha
            self.create_model()

        def get_input(self, s, a):
            # if type(a) == int:
            #     act = [a]
            input = np.zeros(shape = self.state_dims*self.num_actions)
            for i in range(self.state_dims):
                input[a*self.state_dims + i] = s[i]
            return torch.tensor(input)

        def __call__(self, s, a):
            # TODO: implement this method
            self.model.eval()
            input = self.get_input(s, a)
            pred = self.model(input.float())
            return pred.detach().numpy()[0]

        def update(self, alpha, G, s, a):
            # TODO: implement this method
            self.model.train()
            input = self.get_input(s, a)
            pred = self.model(input.float())
            G = torch.tensor(G, dtype=torch.float32)
            loss = 0.5 * self.loss_fn(pred, G)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
