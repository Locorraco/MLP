import pandas as pd
import numpy as np
import sys
from sklearn.utils import shuffle
        
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
        net.Fit(X_train, Y_train, X_validation, Y_validation)
        net.Predict(X_test, Y_test)
            
    """
    def __init__(self, INPUT_SIZE=5, OUTPUT_SIZE=5, LAYER_SIZES=[2, 3, 4], SEED=42, LEARNING_RATE=0.001):
        self.seed = SEED
        self.n = len(LAYER_SIZES)
        self.learning_rate = LEARNING_RATE
        self.output_size = OUTPUT_SIZE
        self.layer_sizes = LAYER_SIZES
        self.input_size = INPUT_SIZE
        self.Initialize_weights()
        self.Grad_zero()
        self.Z = []
        self.A = []
        
    def Grad_zero(self):
        self.gradients = []
        old = self.input_size
        for x in range(self.n):
            new = self.layer_sizes[x]
            self.gradients.append(np.matrix(np.zeros((old+1, new))))
            old = new
        self.gradients.append(np.matrix(np.zeros((old+1, self.output_size))))
        
    def Initialize_weights(self):
        """
        Returns initial weights with values close to zero with dimesions according to dimensions in layer_sizes
        """
        self.weights = []
        old = self.input_size
        for x in range(self.n):
            new = self.layer_sizes[x]
            self.weights.append(np.matrix((np.random.rand(old+1, new)-0.5)/100))
            old = new
        self.weights.append(np.matrix((np.random.rand(old+1, self.output_size)-0.5)/100))
    
    def Real_label(self, y):
        """
        Given an integer, returns a vector of length output_size with zeros in every entry except for the entry y, which has a 1
        
        Arguments: 
        y: integer denoting which entry has a 1
        
        Returns:
        out: one-hot vector with one on the y-th entry
        """
        label = np.matrix(np.zeros((1,self.output_size)))
        label[0,y] = 1
        return label
    
    def Relu(self, X):
        """
        Applies ReLU function to every element of the input
        
        Arguments: 
        X: numpy vector of real numbers
        
        Returns:
        out: vector of same dimension after element wise ReLU function
        """
        # X>0 gives boolean values, X*0 if negative and X*1 if positive, which gives us the function
        # relu(X)=0 if X <= 0 and relu(X)=X otherwise
        return np.multiply(X, X>0)
    
    def Sigmoid(self, X):
        """
        Takes vector and applies sigmoid function to every element
        
        Arguments: 
        X: numpy vector of real numbers
        
        Returns:
        out: vector of same dimension after element wise sigmoid function
        """
        return 1. / (1. + np.exp(-X))
    
    def D_sigmoid(self, X):
        return np.multiply(X,1-X)
    
    def Predict(self, entry):
        """
        Given an input numpy matrix, it returns the output without modifying current weights
        """
        aux_vector = entry
        for w in self.weights:
            aux_vector = np.insert(aux_vector, [aux_vector.size], [1])
            aux_vector = np.dot(aux_vector, w)
            aux_vector = self.Sigmoid(aux_vector)
        print(aux_vector) #Get Rid Of
        decision = np.argmax(aux_vector)
        return decision

    def Cost(self, labels, prediction):
        return np.sum((prediction-labels)*(prediction-labels).T)
    
    #TODO
    def Forward(self,entry):
        aux_vector = entry
        for w in self.weights:
            aux_vector = np.insert(aux_vector, [aux_vector.size], [1])
            aux_vector = np.dot(aux_vector, w)
            self.Z.append(aux_vector)
            aux_vector = self.Sigmoid(aux_vector)
            self.A.append(aux_vector)
        return aux_vector
    
    #TODO
    def Back(self, guess, label):
        error = (guess - label)
        for x in range(self.n-1,-1,-1):
            
        self.Z = []
        self.A = []
    
    #TODO
    def Fit(self):
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
    def Epoch(self):
        pass
    
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
    #net.Fit(X_train,X_test,Y_train,Y_test)
