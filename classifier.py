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
from torchvision.datasets import MNIST

from pathlib import Path


target_directory = "FashionMNIST"

DATA_DIR = '.'
download_dataset = False

# fashion_mnist_real_train = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset)
# fashion_mnist_test = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset)

# print(len(fashion_mnist_real_train))
# print(len(fashion_mnist_test))

def flatten(inp):
    return inp.reshape(-1)

transform = transforms.Compose([transforms.ToTensor()])

fashion_mnist_real_train = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
fashion_mnist_test = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

fashion_mnist_train, fashion_mnist_validation = data.random_split(fashion_mnist_real_train, (48000, 12000))

train_loader = data.DataLoader(fashion_mnist_train, batch_size = sample_batch_size, shuffle = True, num_workers = 1)
val_loader = data.DataLoader(fashion_mnist_validation, batch_size = len(fashion_mnist_validation), shuffle = False)





# Print the shape of the data and targets
# print(fashion_mnist_real_train.data.shape)
# print(fashion_mnist_real_train.targets.shape)

# print(fashion_mnist_test.data.shape)
# print(fashion_mnist_test.targets.shape)

# print(len(fashion_mnist_train), len(fashion_mnist_validation))

