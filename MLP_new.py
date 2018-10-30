import pandas as pd
import numpy as np
import sys
from sklearn.utils import shuffle

class NeuralNetMLP:
    def __init__(self, INPUT_SIZE=5, OUTPUT_SIZE=5, LAYER_SIZES=[2, 3, 4], SEED=42):
        self.seed = SEED
        self.num_hlayers = len(LAYER_SIZES)
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.layer_sizes = LAYER_SIZES
        self.weights = self.initial_weights()
        
    def initial_weights(self):
        weights = []
        old = self.input_size
        for x in range(self.num_hlayers):
            new = self.layer_sizes[x]
            weights.append(np.random.rand(old, new))
            old = new
        weights.append(np.random.rand(old, self.output_size))
        return weights
    
    def real_label(self, y):
        label = np.zeros((1,self.output_size))
        label[0,y] = 1
        return label
    
    def sigmoid(self, X):
        return 1. / (1. + np.exp(-X))
    
    def predict(self, i):
        pass

    def fit(self, X_train, Y_train):
        pass
    
    def epoch(self):
        pass
    
    def minibatch(self):
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
    #net.fit(X_train,X_test,Y_train,Y_test)
