# =====================================================================
# Date  : 1 Aug 2023
# Title : imitation learning agent
# Creator : Iman Sharifi
# =====================================================================

import torch
from torch import nn
import numpy as np

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# imitation learning network model
class ILAgent(nn.Module):

    # =====================================================================
    # constructor
    #   lr = learning rate
    #   state_size = number of variables in the state
    #   fc1_units  = fc1 units
    #   fc2_units  = fc2 units
    #   n_actions  = number of actions
    def __init__(self, state_size=5, fc1_units=256, fc2_units=256, n_actions=3, weight_file_path='', lr=0.01):
        super(ILAgent, self).__init__()

        self.lr =  lr
        self.state_size = state_size

        # initialize weights
        self.fc1 = nn.Linear(state_size, fc1_units)     # layer 1
        self.fc2 = nn.Linear(fc1_units, fc2_units)      # layer 2
        self.fc3 = nn.Linear(fc2_units, n_actions)      # layer 3

        # opimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # loss function
        self.loss = nn.MSELoss()

        # store losses of each step
        self.losses = []

        # send weights to device
        self.to(device)

        self.weight_file_path = weight_file_path

        # load from  the saved weight file if available
        try:  
            self.load_state_dict(torch.load(weight_file_path))
            print("The best weight file have been loaded.")

        except:
            print("We could not load the best weight file.")

    # =====================================================================
    # forward function
    def forward(self, state, softmax=True):
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        if softmax:
            actions = torch.softmax(self.fc3(x), dim=1)
        else:
            actions = self.fc3(x)

        return actions

    # =====================================================================
    # save weights to the file
    def save_weights(self, file_path):
        self.weight_file_path = file_path
        torch.save(self.state_dict(), file_path)

    def load_weights(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print('Weights have been loaded.')

    # =====================================================================
    # learn method
    def learn(self, state, desired_action, softmax=False):

        self.optimizer.zero_grad()

        # network action
        il_action = self.forward(state, softmax=softmax)

        # actual (real) action tensor
        # desired_action = desired_action_tensor(desired_action, desired_velocity)
        # desired_action = torch.tensor([desired_action],requires_grad=False)

        # fit the policy net
        loss = self.loss(il_action, desired_action).to(device)
        loss.backward()
        self.optimizer.step()

        # save the loss
        self.losses.append(loss.item())
      

