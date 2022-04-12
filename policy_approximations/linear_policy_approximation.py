import numpy as np
import torch
from torch import nn
from collections import OrderedDict

class NeuralNetworkPA(nn.Module):
    def __init__(self, dims, outputs):
        super(NeuralNetworkPA, self).__init__()
        self.model = nn.Sequential(OrderedDict([('fc1', nn.Linear(dims, outputs)),
                                                ('act1', nn.LogSoftmax(dim=0))]))

    def forward(self, x):
        return self.model(x)

class LinearPiApproximation():

    def create_model(self):
        self.model = NeuralNetworkPA(self.state_dims, self.num_actions)
        self.nll_loss = nn.NLLLoss()

    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.alpha = alpha
        self.create_model()

    def __call__(self,s) -> int:
        self.model.eval()
        s = torch.tensor(s)
        pred = self.model(s.float())
        return torch.multinomial(torch.exp(pred),1).item()
        # action_probs = pred.detach().numpy()
        # print("Probabilities : ", action_probs)
        # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        # return action

    def return_gradient(self, s, a):
        self.model.train()
        s = torch.tensor(s)
        log_prob = self.model(s.float())
        print("log_prob: ", log_prob)
        self.model.zero_grad()
        loss = self.nll_loss(torch.unsqueeze(log_prob, 0), torch.tensor([a]))
        loss.backward()
        grad =  self.model.model[0].weight.grad.numpy()
        print("grad: ", grad)
        #self.model.zero_grad()
        return grad

    def update(self, update_vector):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        with torch.no_grad():
            update_vector = torch.tensor(update_vector)
            self.model.model[0].weight += update_vector
            print("Weights : ", self.model.model[0].weight)

