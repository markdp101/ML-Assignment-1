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

class FeedforwardNeuralNetModel (nn.Module):
    # Starting with 1 hidden layer then will compare the best model accuracy based on other hyperparameters 
    # then add another input layer than observe the validation and accuracy performance over training epochs (via graphs) to 
    # understand optimal number of hidden layers.
    def __init__(self, num_hidden_layers, hidden_dim,  hidden_dim_2, probability,):
        super(FeedforwardNeuralNetModel, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        # Input layer
        # Linear function 1: input_dim --> hidden_dim
        self.fc1 = nn.Linear(784, hidden_dim)
        # Non-linearity 1 (Come back and decide on activation function with justification)
        self.relu1 = nn.ReLU()

        # Implementing regularization using a dropout layer.
        self.dropout1 = nn.Dropout(p = probability)

        if (num_hidden_layers == 2):
            self.fc2 = nn.Linear(hidden_dim, hidden_dim_2)
            self.relu2 = nn.ReLU()

            # Monitor this extra dropout layer.
            # Implementing regularization using a dropout layer.
            self.dropout2 = nn.Dropout(p = probability)

            # Output layer
            # Linear function 3 (readout): hidden_dim --> output_dim
            self.fc3 = nn.Linear(hidden_dim_2, 10)
        else:
            # Output layer
            # Linear function 2 (readout): hidden_dim --> output_dim
            self.fc2 = nn.Linear(hidden_dim_2, 10)
    
    def forward(self, x):
        # Flatten input.
        x = x.view(x.size(0), -1)
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # # Linear function 2
        # out = self.fc2(x)
        # # Non-linearity 2
        # out = self.relu2(out)

        if (self.num_hidden_layers == 2):
            out = self.fc2(out)
            # Monitor this extra dropout
            out = self.relu2(out)
            out = self.dropout2(out)
            out = self.fc3(out)
        else:
            # Dropout 1
            out = self.dropout1(out)

            # Linear function 3 (readout)
            out = self.fc2(out)

        return out