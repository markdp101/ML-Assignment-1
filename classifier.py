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
            'batch_size': [64, 128, 256, 512, 1024]
    }

    combination = 0
    validations = []
    epochsTracker = []
    trainingAccuracies = np.zeros(len(params['learning_rate'])*len(params['hidden_dims'])*len(params['probability'])*len(params['batch_size']))
    hyperparameterCombinations = []

    learningRateValidations = []
    learningRateEpochs = []
    learningRates = []
    hiddenDimsList = []
    dropoutsListList = []
    batchSizesListList = []
    numLearningRateIterations = 0
    for learning_rate in params['learning_rate']:
        hiddenDimsValidations = []
        hiddenDimsEpochs = []
        hiddenDims = []
        dropoutsList = []
        batchSizesList = []
        hiddenDimIterations = 0

        for hidden_dims in params['hidden_dims']:
            dropoutValidations = []
            dropoutEpochs = []
            dropouts = []
            batch_sizes = []
            num_dropout_iterations = 0

            for probability in params['probability']:
                batchSizeValidations = []
                batchSizeEpochs = []
                batchSizes = []
                num_batch_iterations = 0
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
                    epochs = list(range(epochs+1))

                    trainingAccuracies[combination] = accuracy
                    hyperparameterCombinations.append((batch_size, probability, hidden_dims, learning_rate))

                    validations.append(validationAccuracy)
                    batchSizeValidations.append(validationAccuracy)
                    batchSizeEpochs.append(epochs)
                    epochsTracker.append(epochs)

                    batchSizes.append(batch_size)

                    plotBatchSizeAccuracy(epochs, validationAccuracy, combination, learning_rate, hidden_dims, probability, batch_size)

                    combination += 1
                
                plotBatchSizeAccuracies(batchSizes, batchSizeEpochs, batchSizeValidations, num_batch_iterations, learning_rate, hidden_dims, probability)

                num_batch_iterations += 1

                dropoutValidations.append(batchSizeValidations)
                dropoutEpochs.append(batchSizeEpochs)
                dropouts.append(probability)
                batch_sizes.append(batchSizes)
            
            plotDropoutAccuracies(batch_sizes, dropouts, dropoutEpochs, dropoutValidations, num_dropout_iterations, learning_rate, hidden_dims)
            num_dropout_iterations += 1

            hiddenDimsValidations.append(dropoutValidations)
            hiddenDimsEpochs.append(dropoutEpochs)
            dropoutsList.append(dropouts)
            hiddenDims.append(hidden_dims)
            batchSizesList.append(batch_sizes)

        plotHiddenDimsAccuracies(batchSizesList, dropoutsList, hiddenDims, hiddenDimsEpochs, hiddenDimsValidations, hiddenDimIterations, learning_rate)
        hiddenDimIterations += 1

        learningRateValidations.append(hiddenDimsValidations)
        learningRateEpochs.append(hiddenDimsEpochs)
        learningRates.append(learning_rate)
        hiddenDimsList.append(hiddenDims)
        dropoutsListList.append(dropoutsList)
        batchSizesListList.append(batchSizesList)

    plotLearningRateAccuracies(batchSizesListList, dropoutsListList, hiddenDimsList, learningRates, learningRateEpochs, learningRateValidations, numLearningRateIterations)
    numLearningRateIterations += 1

    highestAccuracy = np.max(trainingAccuracies)
    index = np.argmax(trainingAccuracies)

    configuration = hyperparameterCombinations[index]

    print("The highest validation accuracy (" + str(highestAccuracy) + ") has the following hyperparameters:")
    print("Batch Size:", str(configuration[0]))
    print("Dropout Probability:", str(configuration[1]))
    print("Number of neurons in consecutive hidden layers:", str(configuration[2]))
    print("Learning rate for CLR Scheduling:", str(configuration[3]))

def plotBatchSizeAccuracy(epochs, validationAccuracies, combination, learning_rate, hidden_dims, probability, batch_size):
    plt.plot(epochs, validationAccuracies[:len(epochs)], label="Validation Accuracy for batch size: " + str(batch_size))
    
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title("Validation Accuracy for batch size " + str(batch_size) + " with hyperparameters (combination " + str(combination) + "): learning rate bounds: " + str(learning_rate) + ", hidden dims: " + str(hidden_dims) + ", probability: " + str(probability))

    plt.savefig("combination " + str(combination) + ".jpeg", dpi=300)

    plt.clf()

def plotBatchSizeAccuracies(batch_size, batchSizeEpochs, batchSizeValidations, batchSizeIteration, learning_rate, hidden_dims, probability):
    for i in range(len(batchSizeValidations)):
        plt.plot(batchSizeEpochs[i], batchSizeValidations[i][:len(batchSizeEpochs)], label="Batch Size = " + str(batch_size[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title("Validation accuracies of batch sizes for dropout of " + str(probability) + " with hyperparameters of: learning rate bounds: " + str(learning_rate) + ", hidden dims: " + str(hidden_dims))

    plt.legend()

    plt.savefig("batchSizes" + str(batchSizeIteration) + ".jpeg", dpi=300)

    plt.clf()

def plotDropoutAccuracies(batch_sizes, dropout, dropoutEpochs, dropoutValidations, dropoutIteration, learning_rate, hidden_dims):
    for i in range(len(dropoutValidations)):
        for j in range(len(dropoutValidations[i])):
            plt.plot(dropoutValidations[i][j], dropoutEpochs[i][j][:len(dropoutValidations[i])], label="Batch Size: " + str(batch_sizes[i][j]) + ", dropout: " + str(dropout[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title("Validation accuracies of dropouts for hidden dims of " + str(hidden_dims) + " with hyperparameters of: learning rate bounds: " + str(learning_rate) + ", hidden dims: " + str(hidden_dims))

    plt.legend()

    plt.savefig("dropout" + str(dropoutIteration) + ".jpeg", dpi=300)

    plt.clf()

def plotHiddenDimsAccuracies(batchSizesList, dropoutsList, hiddenDims, hiddenDimsEpochs, hiddenDimsValidations, hiddenDimIterations, learning_rate):
    for i in range(len(hiddenDimsValidations)):
        for j in range(len(hiddenDimsValidations[i])):
            for k in range(len(hiddenDimsValidations[i][j])):
                plt.plot(hiddenDimsValidations[i][j][k], hiddenDimsEpochs[i][j][k][:len(hiddenDimsValidations[i][j])], label="Batch Size: " + str(batchSizesList[i][j][k]) + ", dropout: " + str(dropoutsList[i][j]) + ", hidden dims: " + str(hiddenDims[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title("Validation Accuracies of hidden dims for learning rate of " + str(learning_rate) + " with hyperparameters of: learning rate bounds: " + str(learning_rate))

    plt.legend()

    plt.savefig("hiddenDims" + hiddenDimIterations + ".jpeg", dpi=300)

    plt.clf()

def plotLearningRateAccuracies(batchSizesListList, dropoutsListList, hiddenDimsList, learningRates, learningRateEpochs, learningRateValidations, numLearningRateIterations):
    for i in range(len(learningRateValidations)):
        for j in range(len(learningRateValidations[i])):
            for k in range(len(learningRateValidations[i][j])):
                for n in range(len(learningRateValidations[i][j][k])):
                    plt.plot(learningRateValidations[i][j][k][n], learningRateEpochs[i][j][k][n][:len(learningRateValidations[i][j][k])], label="Batch Size: " + str(batchSizesListList[i][j][k][n]) + ", dropout: " + str(dropoutsListList[i][j][k]) + ", hidden dims: " + str(hiddenDimsList[i][j]) + ", learning rate bounds: " + str(learningRates[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title("Validation accuracies for learning rates")

    plt.legend()

    plt.savefig("learningRates" + str(numLearningRateIterations) + ".jpeg", dpi=300)

    plt.clf()                

if __name__ == "__main__":
    main()