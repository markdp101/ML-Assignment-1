# Mark Du Preez - DPRMAR021
# CSC3022F 2025
# Machine Learning Assignment 1
# Code adapted from slides and code provided by Francois Meyer (Lecturer)

import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

from PIL import Image

import pickle

from training import trainModel, getTestDataLoader
from feedforwardneuralnetmodel import FeedforwardNeuralNetModel
from optimize import getTestAccuracy

def main():
    classificationModel = None

    print("---------- Fashion MNIST Classifier ---------- ")
    print("Option 1: Train a new model with the same optimal hyperparameters to classify JPEG images (1)")
    print("Option 2: Load up the pre-trained model to classify JPEG images (2)")
    option = int(input(""))

    # Training a new model with the optimal hyperparameters.
    if (option == 1):
        # Obtain the test, val and training datasets from the fashionMNIST dataset.
        testDataset, trainDataset, validationDataset = getDataSets()
        
        # Learning rate bounds for CLR learning.
        baseLR = 5e-08
        maxLR = 5e-3

        # Hidden layers.
        numHiddenLayers = 2
        hiddenLayerDims = (512, 256)

        # Dropout probability.
        dropoutP = 0.5

        # Batch size.
        batchSize = 256

        # Max num epochs without improvement for early stopping.
        numEpochs = 20

        # Momentum for batch normalisation.
        momentum = 0.4

        # Instantiate a model with 3 hidden layers with the following hidden layer dimensions, batch norm momentum and dropout probability.
        model = FeedforwardNeuralNetModel(numHiddenLayers, dropoutP, momentum, hiddenLayerDims[0], hiddenLayerDims[1])

        # Objective function --> Cross Entropy Loss
        cost = torch.nn.CrossEntropyLoss()

        # Optimizer --> Adam optimizer.
        optimizer = optim.Adam(model.parameters())

        # LR scheduler --> Cyclical learning rate.
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=baseLR, max_lr=maxLR)

        # Train the model and return the model, final accuracy, num epochs and list of validation accuracies for each epoch.
        accuracy, epochs, trainingLoss, validationAccuracy, trainedModel = trainModel(model, optimizer, cost, scheduler, trainDataset, validationDataset, batchSize, numEpochs)

        testLoader = getTestDataLoader(testDataset, batchSize)

        getTestAccuracy(trainedModel, testLoader)

        # Store the trained model for later classification.
        classificationModel = trainedModel

    else:
        # Load up the already trained, optimized, ideal model for classification.
        try:
            with open('model.pkl', 'rb') as f:
                classificationModel = pickle.load(f)
                print("Model loaded successfully.")
        except FileNotFoundError:
            print("Error: 'model.pkl' not found.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)),transforms.ToTensor()])

    filePath = input("Please enter a filepath:\n")

    # Load a greyscale image.
    img = Image.open(filePath).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)

    classify(classificationModel, img)

# Classify the image using the model.
def classify(model, image):
    labelOptions = ["T-shirt","Trouser","Pullover","Dress","Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    model.eval()
    with torch.no_grad(): 

        # Get the probabilities for each class of classification for this image.
        outputs = model(image)

        # Extract the index of the class with the highest probability.
        predicted = torch.argmax(outputs, dim=1).item()

    print('Classifier: ' + labelOptions[predicted])

def getDataSets():
    # Load the datasets. Test and training datasets of FASHIONMNIST.

    target_directory = "FashionMNIST"

    DATA_DIR = '.'
    download_dataset = False

    transform = transforms.Compose([transforms.ToTensor()])

    fashion_mnist_real_train = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

    fashion_mnist_train, fashion_mnist_validation = data.random_split(fashion_mnist_real_train, (48000, 12000))

    return fashion_mnist_test, fashion_mnist_train, fashion_mnist_validation

if __name__ == "__main__":
    main()