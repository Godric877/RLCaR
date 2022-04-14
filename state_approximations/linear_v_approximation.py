import numpy as np
import torch
from torch import nn

class NeuralNetworkVA(nn.Module):
    def __init__(self, dims):
        super(NeuralNetworkVA, self).__init__()
        self.model = nn.Linear(dims, 1)

    def forward(self, x):
        return self.model(x)

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class LinearStateApproximation(Baseline):

    def create_model(self):
        self.model = NeuralNetworkVA(self.state_dims)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.alpha, betas=(0.9, 0.999))

    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.state_dims = state_dims
        self.alpha = alpha
        self.create_model()

    def __call__(self,s) -> float:
        self.model.eval()
        s = torch.tensor(s)
        pred = self.model(s.float())
        return pred.detach().numpy()[0]

    def update(self,s,G):
        self.model.train()
        s = torch.tensor(s)
        pred = self.model(s.float())
        G = torch.tensor(G, dtype=torch.float32)
        loss = 0.5 * self.loss_fn(pred, G)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()