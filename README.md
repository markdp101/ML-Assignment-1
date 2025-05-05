To run the main jpeg image classifier for FASHION MNIST using a feed forward neural network, run classifier.py:

    There are 2 modes that the classifier program can run in:
    ---------- Option 1: Training the neural network with the optimized hyperparameters  (determined during investigation optimization.py) and then using it to classify JPEG images or,
    ---------- Option 2: Load up the pre-trained model (trained with the optimal hyperparameters during investigation) and use it to classify jpeg images.

    Example: CHOOSING OPTION 1
    ---------- Fashion MNIST Classifier ---------- 
    Option 1: Train a new model with the same optimal hyperparameters to classify JPEG images (1)
    Option 2: Load up the pre-trained model to classify JPEG images (2)
    1
    Epoch 0 : loss( tensor(1.1181) )
    Epoch 0 : val_acc( tensor(0.8298) )
    New best epoch  0 acc tensor(0.8298)
    Epoch 1 : loss( tensor(0.5001) )
    Epoch 1 : val_acc( tensor(0.8489) )
    New best epoch  1 acc tensor(0.8489)
    Epoch 2 : loss( tensor(0.4393) )
    Epoch 2 : val_acc( tensor(0.8512) )
    New best epoch  2 acc tensor(0.8512)
    Epoch 3 : loss( tensor(0.4113) )
    Epoch 3 : val_acc( tensor(0.8659) )
    New best epoch  3 acc tensor(0.8659)
    ...
    Epoch 59 : loss( tensor(0.1514) )
    Epoch 59 : val_acc( tensor(0.9026) )
    Epoch 60 : loss( tensor(0.1393) )
    Epoch 60 : val_acc( tensor(0.9054) )
    No improvement for 20 epochs
    Test Accuracy of the model on the 10000 test images: 90.16 %
    Please enter a filepath:
    C:\Users\markd\OneDrive\Desktop\ML Assignments\ML Assignment 1\fashion-jpegs\bag.jpg  
    Classifier: Bag

    Example: CHOOSING OPTION 2
    ---------- Fashion MNIST Classifier ---------- 
    Option 1: Train a new model with the same optimal hyperparameters to classify JPEG images (1)
    Option 2: Load up the pre-trained model to classify JPEG images (2)
    2
    Model loaded successfully.
    Please enter a filepath:
    C:\Users\markd\OneDrive\Desktop\ML Assignments\ML Assignment 1\fashion-jpegs\sneaker2.jpg 
    Classifier: Sneaker

    WARNING: Option 1's runtime is pretty long.

To run the program used to find the optimal set of hyperparameters -> cyclical learning rate bounds, batch norm momentum, dropout probability, hidden layer dimensions and batch sizes:
    ---------- Run optimize.py
               optimize.py will also create graphs for validation accuracy vs epochs for all combinations of hyper parameters, for all combinations of batch sizes, dropout probabilities, hidden dims, learning rates and momentum.
               The graphs will be outputted to JPEG images.
               Additionally, the program also saves each trained model to a pickle file for later use if needed preventing the need to retrain models which is computationally expensive.
    WARNING: RUN TIME WILL BE EXTREMELY LONG AS THE GRID SEARCH CONSISTS OF 80 DIFFERENT COMBINATIONS OF HYPERPARAMETERS USED TO TRAIN A MODEL.

Additional Notes: training.py encapsulates all the functions involved in training a neural network and feedforwardneuralnetmodel.py contains the feed-forward neural network class with its appropriate constructor and forward pass function defined.