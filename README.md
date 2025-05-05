To run the main jpeg image classifier for FASHION MNIST using a feed forward neural network, run classifier.py:

    There are 2 modes that the classifier program can run in:
    ---------- Option 1: Training the neural network with the optimized hyperparameters  (determined during investigation optimization.py) and then using it to classify JPEG images or,
    ---------- Option 2: Load up the pre-trained model (trained with the optimal hyperparameters during investigation) and use it to classify jpeg images.

    Example: CHOOSING OPTION 1
    ---------- Fashion MNIST Classifier ---------- 
    Option 1: Train a new model with the same optimal hyperparameters to classify JPEG images (1)
    Option 2: Load up the pre-trained model to classify JPEG images (2)
    1
    New best epoch  0 acc tensor(0.8356)
    New best epoch  1 acc tensor(0.8544)
    New best epoch  3 acc tensor(0.8612)
    New best epoch  4 acc tensor(0.8713)
    New best epoch  6 acc tensor(0.8802)
    New best epoch  11 acc tensor(0.8844)
    New best epoch  12 acc tensor(0.8891)
    New best epoch  13 acc tensor(0.8903)
    New best epoch  14 acc tensor(0.8905)
    New best epoch  15 acc tensor(0.8957)
    New best epoch  16 acc tensor(0.8958)
    New best epoch  17 acc tensor(0.8972)
    New best epoch  18 acc tensor(0.8989)
    New best epoch  19 acc tensor(0.9006)
    New best epoch  20 acc tensor(0.9018)
    New best epoch  22 acc tensor(0.9019)
    New best epoch  36 acc tensor(0.9020)
    New best epoch  37 acc tensor(0.9022)
    New best epoch  38 acc tensor(0.9060)
    New best epoch  40 acc tensor(0.9062)
    New best epoch  41 acc tensor(0.9075)
    New best epoch  60 acc tensor(0.9105)
    New best epoch  63 acc tensor(0.9106)
    New best epoch  66 acc tensor(0.9109)
    New best epoch  84 acc tensor(0.9118)
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
