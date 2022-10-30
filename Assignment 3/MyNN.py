import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    n_classes = np.max(y_true) + 1
    cmatrix = np.zeros((n_classes, n_classes))
    for i in range(y_true.shape[0]):
        cmatrix[y_true[i], y_pred[i]] += 1
    return cmatrix

def one_hot_encoder(y):
    y_int = y.astype(int).reshape(-1,)
    n_values = np.max(y_int) + 1
    return np.eye(n_values)[y_int]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return 1. * (z > 0)

class neuralnetwork:

    def __init__(self, hidden_layers, activation, activation_derivative,learning_rate=0.1, stop_epsilon=1e-6, batch_size=100, max_epochs=1000, adaptive=False, objective_function = 'MSE'):
        
        self.hidden_layers = hidden_layers
        self.training_epochs = 0
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.stop_epsilon = stop_epsilon
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.adaptive = adaptive
        self.objective_function = objective_function
        self.layers = []
        self.weights = []
        self.biases = []
        self.output = []
        
    def init_weights_biases(self):
        self.weights.clear()
        self.biases.clear()
        if self.activation == sigmoid: # Xavier initialization
            for i in range(1,len(self.layers)):
                self.weights.append(np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(6/(self.layers[i-1] + self.layers[i])))
                self.biases.append(np.zeros((self.layers[i], 1)))
        elif self.activation == relu:  # He initialization
            for i in range(1,len(self.layers)):
                self.weights.append(np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2/self.layers[i-1]))
                self.biases.append(np.zeros((self.layers[i], 1)))
                
    def forward_propagation(self, X):
        self.output = [X] # input layer
        for i in range(len(self.weights)-1):
            net = np.dot(self.weights[i], self.output[i]) + self.biases[i]
            self.output.append(self.activation(net))
        net = np.dot(self.weights[-1], self.output[-1]) + self.biases[-1]
        self.output.append(sigmoid(net))
        return self.output[-1]

    def back_propagation(self, X, y):
        m = X.shape[1]
        self.forward_propagation(X)
        delta = []
        if self.objective_function == 'MSE':
            delta.insert(0,(self.output[-1] - y) * sigmoid_derivative(self.output[-1]))
        elif self.objective_function == 'BCE':
            delta.insert(0,self.output[-1] - y)
        
        for i in range(len(self.weights)-2, -1, -1):
            delta.insert(0,np.dot(self.weights[i+1].T, delta[0]) * self.activation_derivative(self.output[i+1]))
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(delta[i], self.output[i].T) / m
            self.biases[i] -= self.learning_rate * np.sum(delta[i], axis=1, keepdims=True) / m

    def predict(self, X, one_hot=False):
        y_pred = self.forward_propagation(X.T).T
        if one_hot:
            return y_pred
        
        return np.argmax(y_pred, axis=1)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def loss_function(self,X,y):
        m = X.shape[0]
        if self.objective_function == 'MSE':
            return np.sum(np.square(self.predict(X,one_hot=True) - y))/(2*m)
        elif self.objective_function == 'BCE':
            y_pred = self.predict(X,one_hot=True)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/m
    
    def confusion_matrix(self,X,y):
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def fit(self, X, y):
        y = one_hot_encoder(y.reshape(-1, 1))
        self.layers = [X.shape[1]] + self.hidden_layers + [y.shape[1]]
        self.init_weights_biases()
        self.training_epochs = 0
        prev_epoch_loss = float('inf')
        m = X.shape[0]
        for epoch in range(self.max_epochs):
            self.training_epochs += 1
            curr_epoch_loss = 0
            if self.adaptive:
                self.learning_rate = self.base_learning_rate / np.sqrt(self.training_epochs)
            for i in range(0, m, self.batch_size):
                self.back_propagation(X[i:i+self.batch_size].T, y[i:i+self.batch_size].T)
                curr_epoch_loss += self.loss_function(X[i:i+self.batch_size],y[i:i+self.batch_size])
            
            curr_epoch_loss /= (m/self.batch_size)
            if self.stop_epsilon >  abs(prev_epoch_loss - curr_epoch_loss):
                break
            prev_epoch_loss = curr_epoch_loss


