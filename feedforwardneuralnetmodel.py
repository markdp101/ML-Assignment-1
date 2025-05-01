# Mark Du Preez - DPRMAR021
# CSC3022F 2025
# Machine Learning Assignment 1
# Code apapted from slides provided by Francois Meyer (Lecturer)

# FASHION MNIST Classifier

# Input layer dimensionality (input_dim) = 784
# Output layer dimensionality (output_dim) = 10

# Number of hidden layers = 2
# Hidden layer dimensionality (hidden_dim) = (Come back after testing)

import torch

import torch.nn as nn

class FeedforwardNeuralNetModel (nn.module):
    # Starting with 1 hidden layer then will compare the best model accuracy based on other hyperparameters 
    # then add another input layer than observe the validation and accuracy performance over training epochs (via graphs) to 
    # understand optimal number of hidden layers.
    def __init__(self, input_dim, hidden_dim, output_dim, probability):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Input layer
        # Linear function 1: input_dim --> hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1 (Come back and decide on activation function with justification)
        self.relu1 = nn.Relu()

        # Implementing regularization using a dropout layer.
        self.dropout1 = nn.Dropout(p = probability)

        # # Hidden layers
        # # Linear function 2: hidden_dim --> hidden_dim
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # # Non-linearity 2 (Come back and decide on activation function with justification)
        # self.relu2 = nn.Relu()

        # Output layer
        # Linear function 2 (readout): hidden_dim --> output_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # # Linear function 2
        # out = self.fc2(x)
        # # Non-linearity 2
        # out = self.relu2(out)

        # Dropout 1
        out = self.dropout1(out)

        # Linear function 3 (readout)
        out = self.fc2(x)
        return out