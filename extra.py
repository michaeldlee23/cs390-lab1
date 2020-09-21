import sys, getopt
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
import random
import matplotlib.pyplot as plt

# Full print numpy
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
NUM_INPUTS = 784

# Net parameters
DATASET = 'mnist'
EPOCHS = 100
HIDDEN_LAYERS = 1
HIDDEN_NEURONS = 20
BATCH_SIZE = 100
ACTIVATION = 'sigmoid'

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"

# Toggle error plotting.
PLOT_ERROR = False

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1, activation = ACTIVATION):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W = list()

        # initialize weights
        self.W.append(np.random.randn(self.inputSize, self.neuronsPerLayer))
        for i in range(0, hiddenLayers - 1):
          self.W.append(np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer))
        self.W.append(np.random.randn(self.neuronsPerLayer, self.outputSize))

        if activation.lower() == 'sigmoid':
          self.activation = self.__sigmoid
          self.activationDerivative = self.__sigmoidDerivative
        elif activation.lower() == 'relu':
          self.activation = self.__relu
          self.activationDerivative = self.__reluDerivative
        else:
          raise ValueError('Activation function not recognized')


    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def __relu(self, x):
        x[x < 0] = 0
        return x

    def __reluDerivative(self, x):
        x[x <= 0] = 0
        x[x  > 0] = 1
        return x

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Plots training and validation error over time
    def __plotLearningCurve(self, error):
        x = [i + 1 for i in range(numEpochs)]
        plt.ylim(0, 1)
        plt.plot(x, error['train'], color = 'blue', label = 'training error')
        plt.axhline(y=error['train'][-1], color='purple', ls='--', label='y=%s' % round(error['train'][-1], 4))
        plt.plot(x, error['valid'], color = 'red', label = 'validation error')
        plt.axhline(y=error['valid'][-1], color = 'orange', ls='--', label='y=%s' % round(error['valid'][-1], 4))
        plt.xlabel('epoch')
        plt.ylabel('mean MSE across all classes')
        plt.title('NN Learning Curve, %s Hidden Neurons' % hiddenNeurons)
        plt.legend()
        plt.savefig('./assets/error-%s-%s-%s.png' % (hiddenLayers, hiddenNeurons, numEpochs))


    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = EPOCHS, minibatches = True, mbs = BATCH_SIZE):
        print('HYPERPARAMETERS:\
              \nnet: %s\tepochs: %s\tbatchSize: %s\thiddenSize: %s\thiddenLayers: %s\tactivation: %s\n' %
              (algorithm, numEpochs, batchSize, hiddenNeurons, hiddenLayers, activationFunction))
        error = {'train': [], 'valid': []}
        for i in range(epochs):
          print("epoch:", i)
          # Split data into training and validation set
          proportion = 0.1
          partition = int(proportion * len(yVals))      # Get partition index
          indices = np.random.permutation(len(yVals))   # Get shuffled indices
          xValsShuffled = xVals[indices, :]             # Shuffle xVals
          yValsShuffled = yVals[indices]                # Shuffle yVals, still aligned with xVals
          xTrain = xValsShuffled[partition:]            # Take 1-proportion entities as training
          yTrain = yValsShuffled[partition:]
          xValid = xValsShuffled[:partition]            # Take proportion entities as validation
          yValid = yValsShuffled[:partition]

          if minibatches == True:
            xBatches, yBatches = self.__batchGenerator(xTrain, mbs), self.__batchGenerator(yTrain, mbs)
            for xBatch, yBatch in zip(xBatches, yBatches):
              delta = self.__backpropagate(xBatch, yBatch)
              for j in range(len(delta)):
                self.W[j] -= self.lr * delta[j]
          else:
            delta = self.__backpropagate(xTrain, yTrain)
            for j in range(len(delta)):
              self.W[j] -= self.lr * delta[j]
          if PLOT_ERROR:
            error['train'].append(self.__calculateError(yTrain, self.predict(xTrain)))
            error['valid'].append(self.__calculateError(yValid, self.predict(xValid)))

        if PLOT_ERROR:
          self.__plotLearningCurve(error)

    # Forward pass.
    def __forward(self, input):
        # i inputs, j hidden neurons, k outputs, n entities
        z, a = list(), list()
        z.append(np.dot(input, self.W[0]))        # n x j matrix representing net output of input layer before activation
        a.append(self.activation(z[0]))           # n x j matrix of activity of first hidden layer
        for i in range(1, len(self.W)):           # continually feed forward through layers
          z.append(np.dot(a[i - 1], self.W[i]))
          a.append(self.activation(z[i]))
                                                  # finally outputs n x k matrix of predicted output
        return z, a

    # Predict.
    def predict(self, xVals):
        _, a = self.__forward(xVals)
        yHat = a[-1]  # Predicted output is last element of activity list
        print(yHat)
        prediction = np.zeros_like(yHat)
        prediction[np.arange(len(yHat)), yHat.argmax(1)] = 1
        print('\n\n\n', prediction)
        return prediction

    # Calculate mean error.
    def __calculateError(self, y, yHat):
      mse = 0.5 * sum((y - yHat) ** 2) / np.shape(y)[0]
      return sum(mse)

    # Back propagate.
    def __backpropagate(self, x, y):
      z, a = self.__forward(x)
      yHat = a[-1]
      n = np.shape(y)[0]

      error, delta = [None] * len(self.W), [None] * len(self.W)
      # Last layer is special case since error has to be calculated
      error[-1] = np.multiply((yHat - y) / n, self.activationDerivative(z[-1]))
      delta[-1] = np.dot(a[-2].T, error[-1])

      # Backpropagate through hidden layers
      for i in range(len(z) - 2, 0, -1):
        error[i] = np.dot(error[i + 1], self.W[i + 1].T) * self.activationDerivative(z[i])
        delta[i] = np.dot(a[i].T, error[i])

      # First layer is special case since input x is not in activity matrix
      error[0] = np.dot(error[1], self.W[1].T) * self.activationDerivative(z[0])
      delta[0] = np.dot(x.T, error[0])

      return delta


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    if dataset == 'mnist':
      mnist = tf.keras.datasets.mnist
      (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif dataset == 'iris':
      iris = load_iris()
      # Break given data into training set and test set
      xVals, yVals = iris.data, iris.target
      proportion = 0.3
      partition = int(proportion * len(yVals))      # Get partition index
      indices = np.random.permutation(len(yVals))   # Get shuffled indices
      xValsShuffled = xVals[indices, :]             # Shuffle xVals
      yValsShuffled = yVals[indices]                # Shuffle yVals, still aligned with xVals
      xTrain = xValsShuffled[partition:]            # Take 1-proportion entities as training
      yTrain = yValsShuffled[partition:]
      xTest = xValsShuffled[:partition]            # Take proportion entities as test
      yTest = yValsShuffled[:partition]
    else:
      raise ValueError('Unrecognized dataset')
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if dataset == 'mnist':
      xTrain, xTest = xTrain / 255.0, xTest / 255.0
      xTrain = xTrain.reshape(np.shape(xTrain)[0], -1)
      xTest = xTest.reshape(np.shape(xTest)[0], -1)

    global NUM_CLASSES
    NUM_CLASSES = len(set(yTrain))  # Probably shouldn't change global constants but ¯\_(ツ)_/¯
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if algorithm == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif algorithm == "custom_net":
        print("Building and training Custom_NN.")
        nn = NeuralNetwork_2Layer(len(xTrain[0]), len(yTrain[1]), hiddenNeurons, activation=activationFunction)
        nn.train(xTrain, yTrain, epochs=numEpochs, mbs=batchSize)
        return nn
    elif algorithm == "tf_net":
        print("Building and training TF_NN.")
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(hiddenNeurons, activation=tf.nn.sigmoid),
                                            tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(xTrain, yTrain, validation_split=0.1, batch_size=batchSize, epochs=numEpochs, shuffle=True)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if algorithm == "guesser":
        return guesserClassifier(data)
    elif algorithm == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif algorithm == "tf_net":
        print("Testing TF_NN.")
        yHat = model.predict(data)
        # Convert prediction into single class output
        prediction = np.zeros_like(yHat)
        prediction[np.arange(len(yHat)), yHat.argmax(1)] = 1
        return prediction
    else:
        raise ValueError("Algorithm not recognized.")


def calcF1Score(precision, recall):
  return 2 * ((precision * recall) / (precision + recall)) if precision > 0 and recall > 0 else 0

def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    # Create confusion matrix
    acc = 0
    confusionMatrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1))
    for i in range(preds.shape[0]):
        predictedValue = np.argmax(preds[i])
        actualValue = np.argmax(yTest[i])
        if np.array_equal(preds[i], yTest[i]):  acc += 1
        confusionMatrix[predictedValue][actualValue] += 1   # Update matrix
        confusionMatrix[predictedValue][NUM_CLASSES] += 1   # Update total predicted count for this digit
        confusionMatrix[NUM_CLASSES][actualValue] += 1      # Update total actual count for this digit
    confusionMatrix[NUM_CLASSES][NUM_CLASSES] = sum(confusionMatrix[NUM_CLASSES])
    accuracy = acc / preds.shape[0]

    # Calculate F1 scores based off of confusion matrix
    f1Matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for r in range(NUM_CLASSES):
      for c in range(NUM_CLASSES):
        if r == c:  # F1 score only makes sense by class
          precision = confusionMatrix[r][c] / confusionMatrix[r][NUM_CLASSES]
          recall = confusionMatrix[r][c] / confusionMatrix[NUM_CLASSES][c]
          f1Matrix[r][c] = calcF1Score(precision, recall)
        else:
          f1Matrix[r][c] = np.nan

    print('HYPERPARAMETERS:\
          \nnet: %s\tepochs: %s\tbatchSize: %s\thiddenSize: %s\thiddenLayers: %s\tactivation: %s\n' %
          (algorithm, numEpochs, batchSize, hiddenNeurons, hiddenLayers, activationFunction))
    np.set_printoptions(formatter={'int': '{.6f}'.format}, precision=4, suppress=True)
    print("Classifier algorithm: %s" % algorithm)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Classifier confusion matrix:\n%s\n" % confusionMatrix)
    print("Classifier F1 score matrix:\n%s" % f1Matrix)
    print()



#=========================<Main>================================================


def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    # Accept CLI args to define hyperparameters
    algorithm = ALGORITHM
    dataset = DATASET
    numEpochs = EPOCHS
    batchSize = BATCH_SIZE
    hiddenNeurons = HIDDEN_NEURONS
    hiddenLayers = HIDDEN_LAYERS
    activationFunction = ACTIVATION
    argv = sys.argv[1:]
    try:
      opts, args = getopt.getopt(argv, 'a:d:e:b:n:f:l:h')
    except:
      raise ValueError('Unrecognized argument')

    for opt, arg in opts:
      try:
        if opt in ['-a']:
          algorithm = arg
          if algorithm != 'custom_net' and algorithm != 'tf_net':
            raise ValueError('Unrecognized algorithm. Try "custom_net" or "tf_net"')
        elif opt in ['-d']:
          dataset = arg
          if dataset != 'mnist' and dataset != 'iris':
            raise ValueError('Unrecognized dataset. Try "mnist" or "iris"')
        elif opt in ['-e']:
          numEpochs = int(arg)
          if numEpochs < 1:
            raise ValueError('Number of epochs must be at least 1')
        elif opt in ['-b']:
          batchSize = int(arg)
          if batchSize < 1:
            raise ValueError('Batch size must be at least 1')
        elif opt in ['-n']:
          hiddenNeurons = int(arg)
          if hiddenNeurons < 1:
            raise ValueError('Number of hidden neurons must be at least 1')
        elif opt in ['-f']:
          activationFunction = arg
          if activationFunction.lower() != 'sigmoid' and activationFunction.lower() != 'relu':
            raise ValueError('Unrecognized activation function. Try "sigmoid" or "relu"')
        elif opt in ['-l']:
          hiddenLayers = int(arg)
          if hiddenLayers < 1:
            raise ValueError('Number of hidden layers must be at least 1')
        elif opt in ['-h']:
          print('Usage:\n\
            \t-a [algorithm | custom_net, tf_net]\n\
            \t-d [dataset | mnist, iris]\n\
            \t-e [number of epochs]\n\
            \t-b [batch size]\n\
            \t-n [number of hidden neurons]\n\
            \t-l [number of hidden layers]\n\
            \t-f [activation function | sigmoid, relu]')
          sys.exit()
      except SystemExit as se:
        raise
      except:
        raise ValueError('Invalid arguments. See -h for help')

    main()


