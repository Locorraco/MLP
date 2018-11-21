import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
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
            self.weights.append(np.matrix((np.random.rand(old+1, new)-0.5)*2))
            old = new
        self.weights.append(np.matrix((np.random.rand(old+1, self.output_size)-0.5)*2))
    
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
        return 1. / (1. + np.exp(np.clip(-X,-250, 250)))
    
    def D_sigmoid(self, X):
        return np.multiply(X,1-X)
    
    def Predict(self, entry):
        """
        Given an input numpy matrix, it returns the output without modifying current weights
        """
        aux_vector = entry
        for w in self.weights:
            aux_vector = np.insert(aux_vector, [aux_vector.size], [1.])
            aux_vector = np.dot(aux_vector, w)
            aux_vector = self.Sigmoid(aux_vector)
        #decision = np.argmax(aux_vector)
        return aux_vector

    def Cost(self, entry, labels):
        prediction = self.Predict(entry)
        #print(rl, end=" ")
        #print(prediction, end=" ")
        #print("hi")
        cost = np.sum((prediction-labels).T*(prediction-labels))
        return cost
    
    def Forward(self,entry):
        aux_vector = entry
        self.entry = entry
        self.A.append(np.insert(aux_vector, [aux_vector.size], [1]))
        for w in self.weights:
            aux_vector = np.insert(aux_vector, [aux_vector.size], [1])
            aux_vector = np.dot(aux_vector, w)
            self.Z.append(aux_vector)
            aux_vector = self.Sigmoid(aux_vector)
            self.A.append(np.insert(aux_vector, [aux_vector.size], [1]))
        return aux_vector
    
    def Back(self, label):
        error = (self.A[-1][:,:-1] - label)
        delta_CA = 2*error
        delta_AZ = self.D_sigmoid(self.Z[-1])
        delta_ZW = self.A[-2]
        np.clip(delta_ZW, -100, 100)
        np.clip(delta_AZ, -100, 100)
        np.clip(delta_CA, -100, 100)
        delta_CA_old = delta_CA
        delta_AZ_old = delta_AZ
        self.gradients[-1] += self.learning_rate*delta_ZW.T*np.multiply(delta_CA, delta_AZ)
        for temp_x in range(self.n-1):
            x = self.n-temp_x
            #check math
            delta_CA = np.multiply(delta_CA_old,delta_AZ_old)*self.weights[x][:-1].T
            delta_AZ = self.D_sigmoid(self.Z[x-1])
            delta_ZW = self.A[x-1]
            np.clip(delta_ZW, -100, 100)
            np.clip(delta_AZ, -100, 100)
            np.clip(delta_CA, -100, 100)
            self.gradients[x-1] += self.learning_rate*(delta_ZW.T*np.multiply(delta_CA, delta_AZ))
            delta_CA_old = delta_CA
            delta_AZ_old = delta_AZ
        
        delta_CA = np.multiply(delta_CA_old,delta_AZ_old)*self.weights[1][:-1].T
        delta_AZ = self.D_sigmoid(self.Z[0])
        delta_ZW = self.A[0].reshape(self.input_size+1,1)
        np.clip(delta_ZW, -100, 100)
        np.clip(delta_AZ, -100, 100)
        np.clip(delta_CA, -100, 100)
        self.gradients[0] += self.learning_rate*np.clip(delta_ZW*np.multiply(delta_CA, delta_AZ),-100,100)
    
    def Validation(self,X_validation, Y_validation):
        cost = 0
        mini = 8
        for minibatch in self.Epoch(X_validation, Y_validation, mini):
            for entry,label in minibatch:
                cost += self.Cost(entry, label)
            cost/=len(minibatch)
        cost/=mini
        print(cost)
        return cost
    
    def Step(self):
        for w in range(self.n+1):
            self.weights[w] += self.gradients[w]
            np.clip(self.weights[w], -10, 10)
        self.Grad_zero()
    
    def Fit(self,X_train, Y_train, X_test, Y_test):
        epoch = 1000
        """
        Trains neural network with two panda dataframes.
        
        Arguments:
            X_train: a pandas dataframe which does not include the label
          np.clip(  Y_train: label for the training data
            
            X_train: pandas dataframe of unlabeled
            Y_train: label for the training data
        """
        for i in range(epoch):
            for minibatch in self.Epoch(X_train, Y_train, 100):
                for entry,label in minibatch:
                    self.Forward(entry)
                    self.Back(label)
                self.Step()
            print("TRAINING: ", end="")
            self.Validation(X_train, Y_train)
            print("TESTING: ", end="")
            self.Validation(X_test, Y_test)
            print()
   
    def Epoch(self, X_train, Y_train, batch_size):
        Z_train = shuffle(pd.concat([X_train, Y_train], axis=1))
        batch = []
        for x in range(len(Z_train)):
            minibatch = []

            for i in range(batch_size):
                entry = []
                arry = []
                
                row = Z_train[i:i+1]
                
                for j in range(len(X_train.columns)):
                    temp = row[j].values[0]
                    arry.append(temp)
                
                self.row = row
                self.X_train = X_train
                last = row[len(X_train.columns)].values[0]
                arr = np.array(arry)
                entry.append(arr)
                entry.append(last)
                
                minibatch.append(entry)
            
            batch.append(minibatch)
        return batch
    
def read_data():
    ''' Loads the Boston housing dataset from the "BostonHousing.data" CSV file
    
    Returns
    -----------
    X : array, shape = (506, 13)
        Original features.
    y : array, shape = (506, 1)
        Target values.

    '''
    dataset = np.loadtxt("housing.data")

    # Split into input and output variables
    X = dataset[:,0:-1]
    y = dataset[:,-1:]
    return X, y

def shuffle_data(X, y):
    ''' Shuffles features and target values of a dataset (keeping associations)
    
    Returns
    -----------
    X : array, shape = (506, 13)
        Randomly shuffled original features.
    y : array, shape = (506, 1)
        Randomly shuffled target values.

    '''
    r = np.random.permutation(len(y))
    return X[r], y[r]

if __name__ == "__main__":
    
    X, y = read_data()
    X, y = shuffle_data(X, y)

    # Normalize dataset and keep scalers for "unscaling"
    sc_x = MinMaxScaler()
    sc_y = MinMaxScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)
    
    X_train = pd.DataFrame(X_std[:400])
    Y_train = pd.DataFrame(y_std[:400])
    Y_train.columns = [13]
    X_test = pd.DataFrame(X_std[400:])
    Y_test = pd.DataFrame(y_std[400:])
    Y_test.columns= [13]
    
    net = NeuralNetMLP(INPUT_SIZE=13, OUTPUT_SIZE=1, LAYER_SIZES = [10,10])
    net.Fit(X_train,Y_train,X_test,Y_test)
 
