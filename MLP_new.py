import numpy as np
import sys

class NeuralNetMLP:
    def __init__(self, mb_size, n_layers, dim_layers, n_epochs, l_rate):

        self.n_layers = n_layers
        self.dim_layers = dim_layers
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.mb_size= mb_size
        
    def sigmoid(self, x):
        return 1 / (1. + np.exp(-x))

    #TODO: modificar
    def forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_layers]
        # -> [n_samples, n_layers]
        value = np.dot(X, self.layer_h) + self.bias_1

        # step 2: activation of hidden layer
        value_a = self.sigmoid(value)

        # step 3: net input of output layer
        # [n_samples, n_layers] dot [n_layers, n_classlabels]
        # -> [n_samples, n_classlabels]

        output = np.dot(value_a, self.weights_2) + self.bias_2

        # step 4: activation output layer
        output_a = self.sigmoid(output)

        return value, value_a, output, output_a
    
    def class_count(Y):
        return len(list(set(Y)))
    
    #TODO: set parameters/ modify
    def backprop():
        # [n_samples, n_classlabels]
        sigma_out = a_out - onehot[batch_idx]

        # [n_samples, n_layers]
        sigmoid_derivative_h = a_h * (1. - a_h)

        # [n_samples, n_classlabels] dot [n_classlabels, n_layers]
        # -> [n_samples, n_layers]
        sigma_h = (np.dot(sigma_out, self.weights_2.T) *
                sigmoid_derivative_h)

        # [n_features, n_samples] dot [n_samples, n_layers]
        # -> [n_features, n_layers]
        grad_weights_1 = np.dot(X_train[batch_idx].T, sigma_h)
        grad_bias_1 = np.sum(sigma_h, axis=0)

        # [n_layers, n_samples] dot [n_samples, n_classlabels]
        # -> [n_layers, n_classlabels]
        grad_weights_2 = np.dot(a_h.T, sigma_out)
        grad_bias_2 = np.sum(sigma_out, axis=0)

        # Regularization and weight updates
        delta_weights_1 = (grad_weights_1 + self.l2*self.weights_1)
        delta_bias_1 = grad_bias_1 # bias is not regularized
        self.weights_1 -= self.eta * delta_weights_1
        self.bias_1 -= self.eta * delta_bias_1

        delta_weights_2 = (grad_weights_2 + self.l2*self.weights_2)
        delta_bias_2 = grad_bias_2  # bias is not regularized
        self.weights_2 -= self.eta * delta_weights_2
        self.bias_2 -= self.eta * delta_bias_2

    #TODO: set parameters/ modify
    def evaluate():
        z_h, a_h, z_out, a_out = self.forward(X_train)
        
        cost = self._compute_cost(y_enc=onehot,
                                    output=a_out)

        y_train_pred = self.predict(X_train)
        y_valid_pred = self.predict(X_valid)

        train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                        X_train.shape[0])
        valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                        X_valid.shape[0])

        sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                            '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                            (epoch_strlen, i+1, self.epochs, cost,
                            train_acc*100, valid_acc*100))
        sys.stderr.flush()

        self.eval_['cost'].append(cost)
        self.eval_['train_acc'].append(train_acc)
        self.eval_['valid_acc'].append(valid_acc)

    def initialize_w():
        self.bias_1 = np.zeros(self.n_hidden)
        self.weight_1 = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.bias_2 = np.zeros(n_output)
        self.weight_2 = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

    #TODO: modify
    def fit(self, X_train, Y_train, X_valid, Y_valid):
        
        n_classes = self.class_count(y_train)
        n_input = X_train.shape[1]

        self.initialize_w()
        
        ##self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        self.confusion = [0.,0.,0.,0.]

        ##onehot = self.make_onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self.forward(X_train[batch_idx])

                #set parameters
                self.backprop()

            # Evaluation after each epoch during training
            # set parameters
            self.evaluate()

        return self
