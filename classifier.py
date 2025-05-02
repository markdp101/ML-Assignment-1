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

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import textwrap
import copy
import pickle

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
            'hidden_dims': [(512, 256), (256, 128),(128, 64)],
            'probability': [0.2, 0.5],
            'batch_size': [256, 512]
    }

    combination = 0
    validations = []
    trainingAccuracies = np.zeros(len(params['learning_rate'])*len(params['hidden_dims'])*len(params['probability'])*len(params['batch_size']))
    hyperparameterCombinations = []

    learningRateValidations = []
    learningRateEpochs = []
    learningRates = []
    hiddenDimsList = []
    dropoutsListList = []
    batchSizesListList = []
    numLearningRateIterations = 0
    hiddenDimIterations = 0
    num_dropout_iterations = 0
    num_batch_iterations = 0

    for learning_rate in params['learning_rate']:
        hiddenDimsValidations = []
        hiddenDimsEpochs = []
        hiddenDims = []
        dropoutsList = []
        batchSizesList = []

        for hidden_dims in params['hidden_dims']:
            dropoutValidations = []
            dropoutEpochs = []
            dropouts = []
            batch_sizes = []        

            for probability in params['probability']:
                batchSizeValidations = []
                batchSizeEpochs = []
                batchSizes = []

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

                    model, accuracy, epochs, trainingLoss, validationAccuracy, trainedModel = trainModel(model, optimizer, cost, scheduler, fashion_mnist_train, fashion_mnist_validation, batch_size)
                    epochs = list(range(epochs+1))

                    trainingAccuracies[combination] = copy.deepcopy(accuracy)
                    hyperparameterCombinations.append(copy.deepcopy((batch_size, probability, hidden_dims, learning_rate)))

                    validations.append(copy.deepcopy(validationAccuracy))
                    batchSizeValidations.append(copy.deepcopy(validationAccuracy))
                    batchSizeEpochs.append(copy.deepcopy(epochs))

                    batchSizes.append(copy.deepcopy(batch_size))

                    plotBatchSizeAccuracy(epochs, validationAccuracy, combination, learning_rate, hidden_dims, probability, batch_size)

                    with open('model' + str(combination) + '.pkl', 'wb') as f:
                        pickle.dump(trainedModel, f)

                    combination += 1
                
                plotBatchSizeAccuracies(batchSizes, batchSizeEpochs, batchSizeValidations, num_batch_iterations, learning_rate, hidden_dims, probability)

                num_batch_iterations += 1

                dropoutValidations.append(copy.deepcopy(batchSizeValidations))
                dropoutEpochs.append(copy.deepcopy(batchSizeEpochs))
                dropouts.append(copy.deepcopy(probability))
                batch_sizes.append(copy.deepcopy(batchSizes))
            
            plotDropoutAccuracies(batch_sizes, dropouts, dropoutEpochs, dropoutValidations, num_dropout_iterations, learning_rate, hidden_dims)
            num_dropout_iterations += 1

            hiddenDimsValidations.append(copy.deepcopy(dropoutValidations))
            hiddenDimsEpochs.append(copy.deepcopy(dropoutEpochs))
            dropoutsList.append(copy.deepcopy(dropouts))
            hiddenDims.append(copy.deepcopy(hidden_dims))
            batchSizesList.append(copy.deepcopy(batch_sizes))

        plotHiddenDimsAccuracies(batchSizesList, dropoutsList, hiddenDims, hiddenDimsEpochs, hiddenDimsValidations, hiddenDimIterations, learning_rate)
        hiddenDimIterations += 1

        learningRateValidations.append(copy.deepcopy(hiddenDimsValidations))
        learningRateEpochs.append(copy.deepcopy(hiddenDimsEpochs))
        learningRates.append(copy.deepcopy(learning_rate))
        hiddenDimsList.append(copy.deepcopy(hiddenDims))
        dropoutsListList.append(copy.deepcopy(dropoutsList))
        batchSizesListList.append(copy.deepcopy(batchSizesList))

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
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, validationAccuracies[:len(epochs)], label="Validation Accuracy for batch size: " + str(batch_size))
    
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    title = "Validation Accuracy for batch size " + str(batch_size) + " with hyperparameters (combination " + str(combination) + "): learning rate bounds: " + str(learning_rate) + ", hidden dims: " + str(hidden_dims) + ", dropout probability: " + str(probability)

    title = "\n".join(textwrap.wrap(title, width=100))

    plt.title(title)

    plt.grid(True)

    plt.savefig("combination " + str(combination) + ".jpeg", dpi=300)

    plt.clf()
    plt.close()

def plotBatchSizeAccuracies(batch_size, batchSizeEpochs, batchSizeValidations, batchSizeIteration, learning_rate, hidden_dims, probability):
    plt.figure(figsize=(10, 6))

    for i in range(len(batchSizeValidations)):
        plt.plot(batchSizeEpochs[i], batchSizeValidations[i][:len(batchSizeEpochs[i])], label="Batch Size = " + str(batch_size[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    title = "Validation accuracies of batch sizes for dropout of " + str(probability) + " with hyperparameters of: learning rate bounds: " + str(learning_rate) + ", hidden dims: " + str(hidden_dims)

    title = "\n".join(textwrap.wrap(title, width=100))

    plt.title(title)

    plt.legend()

    plt.grid(True)

    plt.savefig("batchSizes" + str(batchSizeIteration) + ".jpeg", dpi=300)

    plt.clf()
    plt.close()

def plotDropoutAccuracies(batch_sizes, dropout, dropoutEpochs, dropoutValidations, dropoutIteration, learning_rate, hidden_dims):
    plt.figure(figsize=(10, 6))

    for i in range(len(dropoutValidations)):
        for j in range(len(dropoutValidations[i])):
            plt.plot(dropoutEpochs[i][j], dropoutValidations[i][j][:len(dropoutEpochs[i][j])], label="Batch Size: " + str(batch_sizes[i][j]) + ", dropout: " + str(dropout[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    title = "Validation accuracies of dropouts for hidden dims of " + str(hidden_dims) + " with hyperparameters of: learning rate bounds: " + str(learning_rate) + ", hidden dims: " + str(hidden_dims)

    title = "\n".join(textwrap.wrap(title, width=100))

    plt.title(title)

    plt.legend()

    plt.grid(True)

    plt.savefig("dropout" + str(dropoutIteration) + ".jpeg", dpi=300)

    plt.clf()
    plt.close()

def plotHiddenDimsAccuracies(batchSizesList, dropoutsList, hiddenDims, hiddenDimsEpochs, hiddenDimsValidations, hiddenDimIterations, learning_rate):
    plt.figure(figsize=(10, 6))

    for i in range(len(hiddenDimsValidations)):
        for j in range(len(hiddenDimsValidations[i])):
            for k in range(len(hiddenDimsValidations[i][j])):
                plt.plot(hiddenDimsEpochs[i][j][k], hiddenDimsValidations[i][j][k][:len(hiddenDimsEpochs[i][j][k])], label="Batch Size: " + str(batchSizesList[i][j][k]) + ", dropout: " + str(dropoutsList[i][j]) + ", hidden dims: " + str(hiddenDims[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    title = "Validation Accuracies of hidden dims for learning rate of " + str(learning_rate) + " with hyperparameters of: learning rate bounds: " + str(learning_rate)

    title = "\n".join(textwrap.wrap(title, width=100))

    plt.title(title)

    plt.legend()

    plt.grid(True)

    plt.savefig("hiddenDims" + str(hiddenDimIterations) + ".jpeg", dpi=300)

    plt.clf()
    plt.close()

def plotLearningRateAccuracies(batchSizesListList, dropoutsListList, hiddenDimsList, learningRates, learningRateEpochs, learningRateValidations, numLearningRateIterations):
    plt.figure(figsize=(10, 6))

    for i in range(len(learningRateValidations)):
        for j in range(len(learningRateValidations[i])):
            for k in range(len(learningRateValidations[i][j])):
                for n in range(len(learningRateValidations[i][j][k])):
                    plt.plot(learningRateEpochs[i][j][k][n], learningRateValidations[i][j][k][n][:len(learningRateEpochs[i][j][k][n])], label="Batch Size: " + str(batchSizesListList[i][j][k][n]) + ", dropout: " + str(dropoutsListList[i][j][k]) + ", hidden dims: " + str(hiddenDimsList[i][j]) + ", learning rate bounds: " + str(learningRates[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title("Validation accuracies for learning rates")

    plt.legend()

    plt.savefig("learningRates" + str(numLearningRateIterations) + ".jpeg", dpi=300)

    plt.clf()                
    plt.close()

if __name__ == "__main__":
    main()