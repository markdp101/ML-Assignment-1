# Mark Du Preez - DPRMAR021
# CSC3022F 2025
# Machine Learning Assignment 1
# Code apapted from slides provided by Francois Meyer (Lecturer)

import torch

import torch.nn as nn

class FeedforwardNeuralNetModel (nn.module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function 1: input_dim --> hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1 (Come back and decide on activation function with justification)
        self.relu1 = nn.Relu()

        # Linear function 1: input_dim --> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 1 (Come back and decide on activation function with justification)
        self.relu2 = nn.Relu()

        # Linear function 1: input_dim --> hidden_dim
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 1 (Come back and decide on activation function with justification)
        self.relu3 = nn.Relu()

        # Linear function 4 (readout): hidden_dim --> output_dim
        self.fc4 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(x)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3
        out = self.fc3(x)
        # Non-linearity 3
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(x)
        return out