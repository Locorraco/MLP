import pandas as pd
import numpy as np
import sys
from sklearn.utils import shuffle

class Graph:
    def __init__(self):
        self.A = np.matrix([])
        self.Z = np.matrix([])
        
class Minibatch:
    def __init__(self):
        self.parameters = np.matrix([])
        self.labels = np.matrix([])

class NeuralNetMLP:
    """Creates a Multi-Layered Perceptron

    Structure of Neural Network is declared upon initialization, default parameters can (and should) be changed.
    Parameters can be fitted with fit() and seperated X_train and Y_train test variables
    Validation and Testing should be done with predict(), X_test and Y_test

    Parameters:
        INPUT_SIZE:     Vector size of neural network input
        OUTPUT_SIZE:    Vector size of labels (output)
        LAYER_SIZES:    Number of neurons for each hidden layer, represented in a list of vectors
        SEED:           Seed for random number generator (used for testing)

    Example::

        net = NeuralNetMLP(INPUT_SIZE=10, LAYER_SIZES=[10,5,10]):
        net.fit(X_train, Y_train, X_validation, Y_validation)
        net.predict(X_test, Y_test)
            
    """
    def __init__(self, INPUT_SIZE=5, OUTPUT_SIZE=5, LAYER_SIZES=[2, 3, 4], SEED=42, LEARNING_RATE=0.001):
        self.learning_rate = LEARNING_RATE
        self.seed = SEED
        self.n = len(LAYER_SIZES)
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.layer_sizes = LAYER_SIZES
        self.weights = self.initial_weights()
        
    def initial_weights(self):
        """
        Returns initial weights with values close to zero with dimesions according to dimensions in layer_sizes
        """
        weights = []
        old = self.input_size
        for x in range(self.n):
            new = self.layer_sizes[x]
            weights.append(np.random.rand(old+1, new))
            old = new
        weights.append(np.random.rand(old+1, self.output_size))
        return weights
    
    def real_label(self, y):
        """
        Given an integer, returns a vector of length output_size with zeros in every entry except for the entry y, which has a 1
        
        Arguments: 
        y: integer denoting which entry has a 1
        
        Returns:
        out: one-hot vector with one on the y-th entry
        """
        label = np.zeros((1,self.output_size))
        label[0,y] = 1
        return label
    
    def relu(self, X):
        """
        Applies ReLU function to every element of the input
        
        Arguments: 
        X: numpy vector of real numbers
        
        Returns:
        out: vector of same dimension after element wise ReLU function
        """
        # X>0 gives boolean values, X*0 if negative and X*1 if positive, which gives us the function
        # relu(X)=0 if X <= 0 and relu(X)=X otherwise
        return X*(X>0)
    
    def sigmoid(self, X):
        """
        Takes vector and applies sigmoid function to every element
        
        Arguments: 
        X: numpy vector of real numbers
        
        Returns:
        out: vector of same dimension after element wise sigmoid function
        """
        return 1. / (1. + np.exp(-X))
    
    def d_sigmoid(self, X):
        return np.multiply(X,1-X)
    
    def predict(self, inputs):
        """
        Given an input numpy matrix, it returns the output without modifying current weights
        """
        aux_vector = inputs
        for w in self.weights:
            aux_vector = np.insert(aux_vector, [aux_vector.size], [1])
            aux_vector = np.dot(aux_vector, w)
            aux_vector = self.sigmoid(aux_vector)
        decision = np.argmax(aux_vector)
        return decision

    def cost(self, labels, prediction):
        return np.sum((prediction-labels)**2)
    
    #TODO
    def forward(self, inputs):
        graph = Graph()
        aux_vector = inputs
        for w in self.weights:
            #np.insert(temp,[temp.size], [1])
            aux_vector = np.insert(aux_vector, [aux_vector.size], [1])
            aux_vector = np.dot(aux_vector, w)
            graph.Z.append(aux_vector) #add
            aux_vector = self.sigmoid(aux_vector)
            graph.A.append(aux_vector) #add
        return graph
    
    #TODO
    def back(self, graph, minibatch):
        cost = self.cost(minibatch.labels, graph.A[-1])
        n = self.h
        gradient = []
        #first back
        a_l = graph.A.T[-1]
        a_pre = graph.A.T[-2]
        delta_CA = (minibatch.labels-a_l)*2
        
        z_l = graph.Z.T[-1]
        delta_AZ = d_sigmoid(z_l)
        
        delta_ZA = a_pre.T
        
        gradient.append(np.delta_ZA)
        
        
    
    #TODO
    def fit(self, X_train, Y_train, X_validation, Y_validation):
        """
        Trains neural network with two panda dataframes.
        
        Arguments:
            X_train: a pandas dataframe which does not include the label
            Y_train: label for the training data
            
            X_train: pandas dataframe of unlabeled
            Y_train: label for the training data
        """
        
        for batch in self.epoch(X_train, Y_train):
            for minibatch in batch:
                for entry,label in minibatch
                    self.forward(entry.parameters)
                    self.back(graph, entry)
                self.step()
            self.Validation()

    #TODO    
    def epoch(self,X_train,Y_train):
        #randomize
        #Seperate
        #return seperated
        minibatch = Minibatch()
        minibatch.parameters = X_train
        minibatch.labels = Y_train
        return [[minibatch]]
        
    
def Preprocess(data):
    d = {"Iris-setosa" : 0, "Iris-versicolor" : 1, "Iris-virginica" : 2}
    for i in range(len(data)):
        data.iat[i,4] = d[data.iat[i,4]]

if __name__ == "__main__":
    #iris = shuffle(pd.read_csv("Proyectos/MLP/iris.data.txt", header=None))
    iris = shuffle(pd.read_csv("iris.data.txt", header=None))
    Preprocess(iris)
    X_train = iris.iloc[:100,:4]
    X_test  = iris.iloc[100:,:4]
    Y_train = iris.iloc[:100,4]
    Y_test  = iris.iloc[100:,4]
    
    net = NeuralNetMLP()
    #net.fit(X_train,X_test,Y_train,Y_test)
