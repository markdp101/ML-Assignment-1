# Mark Du Preez - DPRMAR021
# CSC3022F 2025
# Machine Learning Assignment 1
# Code adapted from slides and code provided by Francois Meyer (Lecturer)

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
from torchvision import io

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import textwrap
import copy
import pickle

from training import *
from feedforwardneuralnetmodel import *
def main():
    classificationModel = None
     # Extract the optimized, ideal model for classification.
    try:
        with open('model.pkl', 'rb') as f:
            classificationModel = pickle.load(f)
            print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: 'model.pkl' not found.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")


    filePath = input("Please enter a filepath:\n")

    # Load a greyscale image.
    img = io.read_image(filePath, mode=io.ImageReadMode.GRAY)
    img = img.squeeze()

    

if __name__ == "__main__":
    main()