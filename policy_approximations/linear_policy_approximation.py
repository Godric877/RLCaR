import numpy as np
import torch
from torch import nn
from collections import OrderedDict

class NeuralNetworkPA(nn.Module):
    def __init__(self, dims, outputs):
        super(NeuralNetworkPA, self).__init__()
        self.model = nn.Sequential(OrderedDict([('fc1', nn.Linear(dims, outputs)),
                                                ('act1', nn.Softmax(dim=0))]))

    def forward(self, x):
        return self.model(x)

class LinearPolicyApproximation():

    def create_model(self):
        self.model = NeuralNetworkPA(self.state_dims, self.num_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha, betas=(0.9, 0.999))


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

    def __call__(self, s) -> int:
        self.model.eval()
        s = torch.tensor(s)
        pred = self.model(s.float())
        action_probs = pred.detach().numpy()
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.model.train()
        s = torch.tensor(s)
        pred = self.model(s.float())
        log_prob = torch.log(pred)[a].unsqueeze(0)
        loss = - delta * gamma_t * log_prob
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def return_gradient(self, s, a):
        self.model.train()
        s = torch.tensor(s)
        pred = self.model(s.float())
        self.model.zero_grad()
        log_prob = torch.log(pred)[a].unsqueeze(0)
        log_prob.backward()
        grad =  self.model.model[0].weight.grad.numpy()
        #print("grad: ", grad)
        return grad

    def manual_update(self, update_vector):
        with torch.no_grad():
            update_vector = torch.tensor(update_vector)
            self.model.model[0].weight += update_vector
            #print("Weights : ", self.model.model[0].weight)

