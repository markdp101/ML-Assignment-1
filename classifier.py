# Mark Du Preez - DPRMAR021
# CSC3022F 2025
# Machine Learning Assignment 1
# Code apapted from slides and code provided by Francois Meyer (Lecturer)

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from training import *
from feedforwardneuralnetmodel import *
def main():

    target_directory = "FashionMNIST"

    DATA_DIR = '.'
    download_dataset = False

    def flatten(inp):
        return inp.reshape(-1)

    transform = transforms.Compose([transforms.ToTensor()])

    fashion_mnist_real_train = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

    fashion_mnist_train, fashion_mnist_validation = data.random_split(fashion_mnist_real_train, (48000, 12000))

    params = {
            'learning_rate': [ (1e-4, 5e-3), (1e-3, 5e-3)],
            'hidden_dims': [(512, 256), (256, 128)],
            'probability': [0.2, 0.5],
            'batch_size': [256, 512, 1024]
    }
    for learning_rate in params['learning_rate']:
        for hidden_dims in params['hidden_dims']:
            for probability in params['probability']:
                for batch_size in params['batch_size']:
                    hidden_dim1, hidden_dim2 = hidden_dims
                    baseLearningRate, maxLearningRate = learning_rate

                    print("Hidden dims:", hidden_dims)
                    print("Learning Rates:", learning_rate)
                    print("Dropout Probability:", probability)
                    print("Batch Size:", batch_size)

                    model = FeedforwardNeuralNetModel(2, hidden_dim1, hidden_dim2, probability)
                    cost = torch.nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters())
                    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=baseLearningRate, max_lr=maxLearningRate)

                    model, accuracy, epochs, trainingLoss, validationAccuracy = trainModel(model, optimizer, cost, scheduler, fashion_mnist_train, fashion_mnist_validation, batch_size)

if __name__ == "__main__":
    main()