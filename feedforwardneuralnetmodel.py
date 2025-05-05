# Mark Du Preez - DPRMAR021
# CSC3022F 2025
# Machine Learning Assignment 1
# Code adapted from slides provided by Francois Meyer (Lecturer)

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
    def __init__(self, num_hidden_layers, probability, bnMomentum, hidden_dim,  hidden_dim_2, hidden_dim_3=0):
        super(FeedforwardNeuralNetModel, self).__init__()

        self.num_hidden_layers = num_hidden_layers


        # Input dim ----> Hidden_dim_1
        # Input layer
        # Linear function 1: input_dim --> hidden_dim
        self.fc1 = nn.Linear(784, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=bnMomentum)

        # Non-linearity 1 (Come back and decide on activation function with justification)
        self.relu1 = nn.ReLU()

        # Implementing regularization using a dropout layer.
        self.dropout1 = nn.Dropout(p = probability)

        # Hidden_dim_1 ---> Hidden_dim_2

        self.fc2 = nn.Linear(hidden_dim, hidden_dim_2)

        self.bn2 = nn.BatchNorm1d(hidden_dim_2, momentum=bnMomentum)

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(p = probability)



        # Hidden_dim_2 --> Hidden_dim_3

        if (self.num_hidden_layers >= 3):
            self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
            self.bn3 = nn.BatchNorm1d(hidden_dim_3, momentum=bnMomentum)
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(p=probability)
            self.fc4 = nn.Linear(hidden_dim_3, 10)
        else:
            self.fc3 = nn.Linear(hidden_dim_2, 10)
    
    def forward(self, x):
        # Flatten input.
        x = x.view(x.size(0), -1)
        # Linear function 1
        out = self.fc1(x)

        out = self.bn1(out)
        
        # Non-linearity 1
        out = self.relu1(out)

        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if (self.num_hidden_layers >= 3):
            out = self.fc3(out)
            out = self.bn3(out)
            out = self.relu3(out)
            out = self.dropout3(out)
            out = self.fc4(out)
        else:
            out = self.fc3(out)

        return out