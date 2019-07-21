import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_c1_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_c1_2 = nn.Linear(400, 300)
        self.layer_c1_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_c2_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_c2_2 = nn.Linear(400, 300)
        self.layer_c2_3 = nn.Linear(300, 1)

    def forward(self, x, u):
        #concat State and Action
        sa = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        c1 = F.relu(self.layer_c1_1(sa))
        c1 = F.relu(self.layer_c1_2(c1))
        c1 = self.layer_c1_3(c1)
        # Forward-Propagation on the second Critic Neural Network
        c2 = F.relu(self.layer_c2_1(sa))
        c2 = F.relu(self.layer_c2_2(c2))
        c2 = self.layer_c2_3(c2)
        return c1, c2

    #the prediction of first Critic Network, this is used for gradient policy of Actor network.
    def Q1(self, x, u):
        sa = torch.cat([x, u], 1)
        c1 = F.relu(self.layer_c1_1(sa))
        c1 = F.relu(self.layer_c1_2(c1))
        c1 = self.layer_c1_3(c1)
        return c1