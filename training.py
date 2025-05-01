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

def loadAndProcessData ():
    # Load the fashionMNIST dataset and split into training, validation and test sets.
    target_directory = "FashionMNIST"
    DATA_DIR = '.'
    download_dataset = False

    transform = transforms.Compose([transforms.ToTensor()])

    fashion_mnist_real_train = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

    fashion_mnist_train, fashion_mnist_validation = data.random_split(fashion_mnist_real_train, (48000, 12000))

    return fashion_mnist_train, fashion_mnist_validation, fashion_mnist_test

# Helper function to compute accuracy.
def compute_acc(logits, expected):
    pred = logits.argmax(dim=1)
    return (pred == expected).type(torch.float).mean()

def trainModel(model, opt, cost, scheduler, mnist_train, mnist_validation, batchSize):
    # Required hyperparameters:
    # --------------------------Optimizer-------------------------------> Adam Optimizer
    # --------------------------Cost function/Loss function-------------> Cross-Entropy Loss
    # --------------------------Batch size------------------------------> Variable (while optimising)
    # --------------------------CLR (Cyclical Learning Rate scheduler)--> base_lr and max_lr variable (while optimising)

    # Implements early stopping if validation accuracy stops decreasing.

    # Used to keep track of training loss (want to minimize) and validation accuracy (want to maximise).
    train_loss = []
    validation_acc = []

    # Used to store the best model, accuracy and number of epochs.
    best_model = None
    best_acc = None
    best_epoch = None

    # The highest number of epochs if early stopping is not triggered.
    max_epoch = 100

    # The maximum number of epochs permitted with no improvement in validation accuracy.
    no_improvement = 5

    for n_epoch in range(max_epoch):
        model.train()
        loader = data.DataLoader(mnist_train, batch_size=batchSize, shuffle=True, num_workers=1)
        epoch_loss = []
        for X_batch, y_batch in loader:
            opt.zero_grad()
            logits = model(X_batch)
            loss = cost(logits, y_batch)
            loss.backward()
            opt.step()
            scheduler.step()
            epoch_loss.append(loss.detach())
        train_loss.append(torch.tensor(epoch_loss).mean())
        model.eval()
        loader = data.DataLoader(mnist_validation, batch_size=len(mnist_validation), shuffle=False)
        X, y = next(iter(loader))
        logits = model(X)
        acc = compute_acc(logits, y).detach()
        validation_acc.append(acc)
        if best_acc is None or acc > best_acc:
            print("New best epoch ", n_epoch, "acc", acc)
            best_acc = acc
            best_model = model.state_dict()
            best_epoch = n_epoch
        if best_epoch + no_improvement <= n_epoch:
            print("No improvement for", no_improvement, "epochs")
            break

    model.load_state_dict(best_model)

    return model, best_acc, best_epoch, train_loss, validation_acc

def flatten(inp):
    return inp.reshape(-1)