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

def loadAndProcessData (sample_batch_size):
    target_directory = "FashionMNIST"
    DATA_DIR = '.'
    download_dataset = False

    transform = transforms.Compose([transforms.ToTensor()])

    fashion_mnist_real_train = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

    fashion_mnist_train, fashion_mnist_validation = data.random_split(fashion_mnist_real_train, (48000, 12000))

    train_loader = data.DataLoader(fashion_mnist_train, batch_size = sample_batch_size, shuffle = True, num_workers = 1)
    val_loader = data.DataLoader(fashion_mnist_validation, batch_size = sample_batch_size, shuffle = False)
    test_loader = data.DataLoader(fashion_mnist_test, batch_size = sample_batch_size, shuffle = False)


    return train_loader, val_loader, test_loader

def flatten(inp):
    return inp.reshape(-1)