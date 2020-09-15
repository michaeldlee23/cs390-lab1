
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import matplotlib.pyplot as plt


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1, activation = 'sigmoid'):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
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
        #return x * (1 - x)  # assume that the inputted x is already activated

    def __relu(self, x):
        x[x < 0] = 0
        return x

    def __reluDerivative(self, x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        #error = list()
        for i in range(epochs):
          print("epoch:", i)
          if minibatches == True:
            xBatches, yBatches = self.__batchGenerator(xVals, mbs), self.__batchGenerator(yVals, mbs)
            for xBatch, yBatch in zip(xBatches, yBatches):
              (l1d, l2d) = self.__backpropagate(xBatch, yBatch)
              self.W1 -= self.lr * l1d
              self.W2 -= self.lr * l2d
          #print(self.__calculateError(yVals, self.predict(xVals)))
          #error.append(self.__calculateError(yVals, self.predict(xVals)))
        #x = [i for i in range(epochs)]
        #plt.plot(x, error)
        #plt.show()

    # Forward pass.
    def __forward(self, input):
        z2 = np.dot(input, self.W1) # n x j matrix representing net output of layer 1 before fed into sigmoid
        a2 = self.activation(z2)     # n x j matrix of activity of layer 2

        z3 = np.dot(a2, self.W2)    # n x k matrix representing net output of layer 2 before fed into sigmoid
        a3 = self.activation(z3)     # n x k matrix of activity of layer 3 (predicted output)
        return z2, a2, z3, a3

    # Predict.
    def predict(self, xVals):
        _, _, _, yHat = self.__forward(xVals)
        prediction = np.zeros_like(yHat)
        prediction[np.arange(len(yHat)), yHat.argmax(1)] = 1
        return prediction

    # Calculate error.
    def __calculateError(self, y, yHat):
      return 0.5 * sum((y - yHat) ** 2) / np.shape(y)[0]

    # Back propagate.
    def __backpropagate(self, x, y):
      z2, a2, z3, yHat = self.__forward(x)
      n = np.shape(y)[0]

      l2e = np.multiply((yHat - y) / n, self.__sigmoidDerivative(z3))
      #l2e = np.multiply((yHat - y) / n, self.activationDerivative(yHat))
      l2d = np.dot(a2.T, l2e)

      #l1e = np.multiply(np.dot(l2e, self.W2.T), self.__sigmoidDerivative(z2))
      l1e = np.dot(l2e, self.W2.T) * self.__sigmoidDerivative(z2)
      #l1e = np.dot(l2e, self.W2.T) * self.activationDerivative(a2)
      l1d = np.dot(x.T, l1e)

      return (l1d, l2d)


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
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain, xTest = xTrain / 255.0, xTest / 255.0
    xTrain = xTrain.reshape(np.shape(xTrain)[0], -1)
    xTest = xTest.reshape(np.shape(xTest)[0], -1)
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    EPOCHS = 50
    HIDDEN_NEURONS = 20
    BATCH_SIZE = 100
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        #nn = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, HIDDEN_NEURONS, activation='ReLU')
        nn = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, HIDDEN_NEURONS)
        nn.train(xTrain, yTrain, EPOCHS)
        return nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(HIDDEN_NEURONS, activation=tf.nn.sigmoid),
                                            tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.sigmoid)])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(xTrain, yTrain, batch_size=BATCH_SIZE, epochs=EPOCHS)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        yHat = model.predict(data)
        prediction = np.zeros_like(yHat)
        prediction[np.arange(len(yHat)), yHat.argmax(1)] = 1
        return prediction
    else:
        raise ValueError("Algorithm not recognized.")


def calcF1Score(precision, recall):
  return 2 * ((precision * recall) / (precision + recall)) if precision > 0 and recall > 0 else 0

def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    confusionMatrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1))
    for i in range(preds.shape[0]):
        predictedValue = np.argmax(preds[i])
        actualValue = np.argmax(yTest[i])
        if np.array_equal(preds[i], yTest[i]):  acc += 1
        confusionMatrix[predictedValue][actualValue] += 1   # Update matrix
        confusionMatrix[predictedValue][NUM_CLASSES] += 1   # Update total predicted count for this digit
        confusionMatrix[NUM_CLASSES][actualValue] += 1      # Update total actual count for this digit
    accuracy = acc / preds.shape[0]
    f1Matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for r in range(NUM_CLASSES):
      for c in range(NUM_CLASSES):
        if r == c:
          precision = confusionMatrix[r][c] / confusionMatrix[r][NUM_CLASSES]
          recall = confusionMatrix[r][c] / confusionMatrix[NUM_CLASSES][c]
          f1Matrix[r][c] = calcF1Score(precision, recall)
        else:
          f1Matrix[r][c] = np.nan
    np.set_printoptions(formatter={'int': '{.6f}'.format}, precision=4, suppress=True)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Classifier confusion matrix:\n", confusionMatrix)
    print("Classifier F1 score matrix:\n", f1Matrix)
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
