# Mark Du Preez - DPRMAR021
# CSC3022F 2025
# Machine Learning Assignment 1
# Code adapted from slides and code provided by Francois Meyer (Lecturer)
# Used to perform a grid search and find the optimal set of hyperparameters based on the highest test accuracy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets

import torchvision.transforms as transforms

import textwrap
import copy
import pickle

from training import *
from feedforwardneuralnetmodel import *

def main():
    # Load the datasets. Test and training datasets of FASHIONMNIST.

    target_directory = "FashionMNIST"

    DATA_DIR = '.'
    download_dataset = False

    transform = transforms.Compose([transforms.ToTensor()])

    fashion_mnist_real_train = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

    fashion_mnist_train, fashion_mnist_validation = data.random_split(fashion_mnist_real_train, (48000, 12000))

    # Hyperparameters for Grid search
    params = {
            'hidden_dims': [(512, 256)],
            'learning_rate': [(1e-9, 5e-4), (2.5e-8, 5e-3), (5e-8, 5e-3), (1e-7, 5), (2.5e-4, 5e-3)],
            'probability': [0.2, 0.3, 0.4, 0.5],
            'batch_size': [(256, 20)],
            'momentum': [0.3, 0.2, 0.1, 0.4]
    }

    # Counters and lists used to save training data from each combination 

    combination = 0
    validations = []
    trainingAccuracies = []
    testAccuracies = []
    hyperparameterCombinations = []

    momentumValidations = []
    momentumEpochs = []
    combinationsListListListList = []

    numMomentumIterations = 0
    numLearningRateIterations = 0
    hiddenDimIterations = 0
    num_dropout_iterations = 0
    num_batch_iterations = 0

    # Nested for loop for iterating through each combination of hyperparameters.
    for momentum in params['momentum']:
        learningRateValidations = []
        learningRateEpochs = []
        combinationsListListList = []

        for learning_rate in params['learning_rate']:
            hiddenDimsValidations = []
            hiddenDimsEpochs = []
            combinationsListList = []

            for hidden_dims in params['hidden_dims']:
                dropoutValidations = []
                dropoutEpochs = []
                combinationsList = []     

                for probability in params['probability']:
                    batchSizeValidations = []
                    batchSizeEpochs = []
                    combinations = []

                    for batchSize in params['batch_size']:
                        baseLearningRate, maxLearningRate = learning_rate

                        # Print the hyperparameters of current combination to terminal.

                        print("Hidden dims:", str(hidden_dims))
                        print("Learning Rates:", str(learning_rate))
                        print("Dropout Probability:", str(probability))
                        print("Batch Size:", str(batchSize))
                        print("Momentum:", str(momentum))

                        # Extract the batch size and max number of epochs without improvement.
                        batch_size, improvementEpochs= batchSize

                        # Different initializations based on number of hidden layers.
                        if (len(hidden_dims)==3):
                            model = FeedforwardNeuralNetModel(3, probability, momentum, hidden_dims[0], hidden_dims[1], hidden_dims[2])
                        else:
                            model = FeedforwardNeuralNetModel(2, probability, momentum, hidden_dims[0], hidden_dims[1])

                        # Objective function --> Cross Entropy Loss
                        cost = torch.nn.CrossEntropyLoss()

                        # Optimizer --> Adam optimizer.
                        optimizer = optim.Adam(model.parameters())

                        # LR scheduler --> Cyclical learning rate.
                        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=baseLearningRate, max_lr=maxLearningRate)

                        # Train the model and return the model, final accuracy, num epochs and list of validation accuracies for each epoch.
                        accuracy, epochs, trainingLoss, validationAccuracy, trainedModel = trainModel(model, optimizer, cost, scheduler, fashion_mnist_train, fashion_mnist_validation, batch_size, improvementEpochs)

                        # Initialize data loader of test dataset for computing test accuracy.
                        test_loader = getTestDataLoader(fashion_mnist_test, batch_size)
                        testAccuracies.append(getTestAccuracy(trainedModel, test_loader))

                        # For graphing purposes (to include the last epoch)
                        epochs = list(range(epochs+1))

                        # Record the training accuracies and hyperparameter combinations.
                        trainingAccuracies.append(copy.deepcopy(accuracy))
                        hyperparameterCombinations.append(copy.deepcopy((batch_size, probability, hidden_dims, learning_rate, momentum)))

                        validations.append(copy.deepcopy(validationAccuracy))
                        batchSizeValidations.append(copy.deepcopy(validationAccuracy))
                        batchSizeEpochs.append(copy.deepcopy(epochs))

                        combinations.append(copy.deepcopy(combination))

                        # Plot the validation accuracy vs number of epochs for a specific combination.
                        plotBatchSizeAccuracy(epochs, validationAccuracy, combination, momentum, learning_rate, hidden_dims, probability, batch_size)

                        # Store the model to a pickle file to be used later.
                        with open('model' + str(combination) + '.pkl', 'wb') as f:
                            pickle.dump(trainedModel, f)

                        combination += 1
                    
                    # Plot the accuracies for different batch sizes.
                    plotBatchSizeAccuracies(batchSizeEpochs, batchSizeValidations, num_batch_iterations, momentum, learning_rate, hidden_dims, probability, combinations)

                    num_batch_iterations += 1

                    dropoutValidations.append(copy.deepcopy(batchSizeValidations))
                    dropoutEpochs.append(copy.deepcopy(batchSizeEpochs))
                    combinationsList.append(copy.deepcopy(combinations))

                # Plot the accuracies for different dropout probabilities.
                plotDropoutAccuracies(dropoutEpochs, dropoutValidations, num_dropout_iterations, momentum, learning_rate, hidden_dims, combinationsList)
                num_dropout_iterations += 1

                hiddenDimsValidations.append(copy.deepcopy(dropoutValidations))
                hiddenDimsEpochs.append(copy.deepcopy(dropoutEpochs))
                combinationsListList.append(copy.deepcopy(combinationsList))
            
            # Plot the accuracies for different hidden layer configurations.
            plotHiddenDimsAccuracies(hiddenDimsEpochs, hiddenDimsValidations, hiddenDimIterations, momentum, learning_rate, combinationsListList)
            hiddenDimIterations += 1

            learningRateValidations.append(copy.deepcopy(hiddenDimsValidations))
            learningRateEpochs.append(copy.deepcopy(hiddenDimsEpochs))
            combinationsListListList.append(copy.deepcopy(combinationsListList))

        # Plot the accuracies for different learning rates.
        plotLearningRateAccuracies(learningRateEpochs, learningRateValidations, numLearningRateIterations, momentum, combinationsListListList)
        numLearningRateIterations += 1

        momentumValidations.append(copy.deepcopy(learningRateValidations))
        momentumEpochs.append(copy.deepcopy(learningRateEpochs))
        combinationsListListListList.append(copy.deepcopy(combinationsListListList))

        numMomentumIterations += 1

    # Plot the accuracies for different batch norm momentums.
    plotMomentumAccuracies(momentumEpochs, momentumValidations, numMomentumIterations, combinationsListListListList)

    # Obtain the combination with the highest test accuracy and output the hyperparameters for that combination.
    highestAccuracy = np.max(testAccuracies)
    index = np.argmax(testAccuracies)

    configuration = hyperparameterCombinations[index]

    print("The highest validation accuracy (" + str(highestAccuracy) + ") has the following hyperparameters:")
    print("Batch Size:", str(configuration[0]))
    print("Dropout Probability:", str(configuration[1]))
    print("Number of neurons in consecutive hidden layers:", str(configuration[2]))
    print("Learning rate for CLR Scheduling:", str(configuration[3]))
    print("Momentum:", str(configuration[4]))
    print("Test Accuracy:", str(testAccuracies[index]))

# Used to plot the accuracies for a specific hyperparameter combination.
def plotBatchSizeAccuracy(epochs, validationAccuracies, combination, momentum, learning_rate, hidden_dims, probability, batch_size):
    plt.figure(figsize=(10, 6))
    plt.grid(True)

    plt.plot(epochs, validationAccuracies[:len(epochs)], label="Validation Accuracy for batch size: " + str(batch_size))
    
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    title = "Validation Accuracy for batch size " + str(batch_size) + " with hyperparameters: momentum: " + str(momentum) + ", learning rate bounds: " + str(learning_rate) + ", hidden dims: " + str(hidden_dims) + ", dropout probability: " + str(probability)

    title = "\n".join(textwrap.wrap(title, width=100))

    plt.title(title)

    plt.savefig("combination " + str(combination) + ".jpeg", dpi=300)

    plt.clf()
    plt.close()

# Used to plot the accuracies for different batch sizes.
def plotBatchSizeAccuracies(batchSizeEpochs, batchSizeValidations, batchSizeIteration, momentum, learning_rate, hidden_dims, probability, combination):
    plt.figure(figsize=(12, 8))
    plt.grid(True)

    for i in range(len(batchSizeValidations)):
        plt.plot(batchSizeEpochs[i], batchSizeValidations[i][:len(batchSizeEpochs[i])], label="Combination: " + str(combination[i]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    title = "Validation accuracies of batch sizes for dropout of " + str(probability) + " with hyperparameters of: momentum: " + str(momentum) + ", learning rate bounds: " + str(learning_rate) + ", hidden dims: " + str(hidden_dims)

    title = "\n".join(textwrap.wrap(title, width=100))

    plt.title(title)

    plt.legend()

    plt.savefig("batchSizes" + str(batchSizeIteration) + ".jpeg", dpi=300)

    plt.clf()
    plt.close()

# Used to plot the accuracies for different dropout probabilities.
def plotDropoutAccuracies(dropoutEpochs, dropoutValidations, dropoutIteration, momentum, learning_rate, hidden_dims, combination):
    plt.figure(figsize=(12, 8))
    plt.grid(True)

    for i in range(len(dropoutValidations)):
        for j in range(len(dropoutValidations[i])):
            plt.plot(dropoutEpochs[i][j], dropoutValidations[i][j][:len(dropoutEpochs[i][j])], label="Combination: " + str(combination[i][j]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    title = "Validation accuracies of dropouts for hidden_dims of " + str(hidden_dims) + " with hyperparameters of: momentum: " + str(momentum) + ", learning rate bounds: " + str(learning_rate)

    title = "\n".join(textwrap.wrap(title, width=100))

    plt.title(title)

    plt.legend()

    plt.savefig("dropout" + str(dropoutIteration) + ".jpeg", dpi=300)

    plt.clf()
    plt.close()

# Used to plot the accuracies for different hidden layer configurations.
def plotHiddenDimsAccuracies(hiddenDimsEpochs, hiddenDimsValidations, hiddenDimIterations, momentum, learning_rate, combination):
    plt.figure(figsize=(12, 8))
    plt.grid(True)

    for i in range(len(hiddenDimsValidations)):
        for j in range(len(hiddenDimsValidations[i])):
            for k in range(len(hiddenDimsValidations[i][j])):
                plt.plot(hiddenDimsEpochs[i][j][k], hiddenDimsValidations[i][j][k][:len(hiddenDimsEpochs[i][j][k])], label="Combination: " + str(combination[i][j][k]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    title = "Validation Accuracies of hidden dims for learning rate of " + str(learning_rate) + " with hyperparameters of: " + str(momentum) + ", learning rate bounds: " + str(learning_rate)

    title = "\n".join(textwrap.wrap(title, width=100))

    plt.title(title)

    plt.legend()

    plt.savefig("hiddenDims" + str(hiddenDimIterations) + ".jpeg", dpi=300)

    plt.clf()
    plt.close()

# Used to plot the accuracies for different learning rates.
def plotLearningRateAccuracies(learningRateEpochs, learningRateValidations, numLearningRateIterations, momentum, combination):
    plt.figure(figsize=(12, 8))
    plt.grid(True)

    for i in range(len(learningRateValidations)):
        for j in range(len(learningRateValidations[i])):
            for k in range(len(learningRateValidations[i][j])):
                for n in range(len(learningRateValidations[i][j][k])):
                    plt.plot(learningRateEpochs[i][j][k][n], learningRateValidations[i][j][k][n][:len(learningRateEpochs[i][j][k][n])], label="Combination: " + str(combination[i][j][k][n]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title("Validation accuracies of learning rates for momentum of " + str(momentum))

    plt.legend()

    plt.savefig("learningRates" + str(numLearningRateIterations) + ".jpeg", dpi=300)

    plt.clf()                
    plt.close()

# Used to plot the accuracies for different momentums for batch norm.
def plotMomentumAccuracies(momentumEpochs, momentumValidations, numMomentumIterations, combination):
    plt.figure(figsize=(12, 8))

    for i in range(len(momentumValidations)):
        for j in range(len(momentumValidations[i])):
            for k in range(len(momentumValidations[i][j])):
                for n in range(len(momentumValidations[i][j][k])):
                    for z in range(len(momentumValidations[i][j][k][n])):
                        plt.plot(momentumEpochs[i][j][k][n][z], momentumValidations[i][j][k][n][z][:len(momentumEpochs[i][j][k][n][z])], label="Combination: " + str(combination[i][j][k][n][z]))

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")

    plt.title("Validation accuracies for momentums")

    plt.legend()

    plt.savefig("momentums" + str(numMomentumIterations) + ".jpeg", dpi=300)

    plt.clf()                
    plt.close()

# Compute the test accuracy for a particular model.
def getTestAccuracy(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # Get probabilities for each class
            outputs = model(images)

            # Get the class with the highest probability
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)

            # Compare predicted classes with true labels
            correct += (predicted == labels).sum().item()  

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))    

    return (100 * correct / total)

if __name__ == "__main__":
    main()