import numpy as np
import scipy
import pandas as pd
import tqdm
import sys
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def confusion_matrix(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    n_classes = np.max(y_true) + 1
    cmatrix = np.zeros((n_classes, n_classes),dtype=int)
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
    if z > 0:
        return 1.0
    else:
        return 0.0

class neuralnetwork:

    def __init__(self, hidden_layers, activation, activation_derivative,learning_rate=0.1, stop_epsilon=1e-6, batch_size=100, max_epochs=10, adaptive=False, objective_function = 'MSE'):
        
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

def dataloader(train,test):
    training_data = pd.read_csv(f'{train}/fmnist_train.csv',header=None).to_numpy(dtype = 'float')
    x_train, y_train = training_data[:, :-1]/255, training_data[:, -1]


    test_data = pd.read_csv(f'{test}/fmnist_test.csv',header  = None).to_numpy(dtype = 'float')
    x_test, y_test = test_data[:, :-1]/255, test_data[:, -1]
    return x_train, y_train, x_test, y_test

def main():
    train_loc = sys.argv[1]
    test_loc = sys.argv[2]
    part = sys.argv[4]
    output_folder = sys.argv[3]
    output_file_path = f'{output_folder}/{part}.txt'
    output_file = open(output_file_path,"w")
    x_train, y_train, x_test, y_test = dataloader(train_loc,test_loc)

    if part == 'b' :
        nns = []
        traintime = []
        hidden_layer_units = [5,10,15,20,25]
        for n in tqdm.tqdm(hidden_layer_units):
            
            nnb = neuralnetwork(hidden_layers=[n],activation=sigmoid,activation_derivative=sigmoid_derivative)
            start = time.time()
            nnb.fit(x_train,y_train)
            end = time.time()
            traintime.append(end-start)   
            nns.append(nnb)  
        train_scores = []
        test_scores = []
        for nn in nns:
            train_scores.append(nn.score(x_train,y_train))
            test_scores.append(nn.score(x_test,y_test))
        fig,ax = plt.subplots()
        ax.plot(hidden_layer_units,train_scores,label = 'train')
        ax.plot(hidden_layer_units,test_scores,label = 'test')
        ax.set_xlabel('Number of hidden units')
        ax.set_ylabel('Accuracy')
        ax.legend()
        plt.savefig(f'{output_folder}/{part}_accuracy.png')
        plt.close()
        fig,ax = plt.subplots()
        ax.plot(hidden_layer_units,traintime)
        ax.set_xlabel('Number of hidden units')
        ax.set_ylabel('Training time')
        plt.savefig(f'{output_folder}/{part}_training_time.png')
        plt.close()
        for n in range(len(hidden_layer_units)):
            output_file.write(f'Hidden units: {hidden_layer_units[n]}\nTrain accuracy: {train_scores[n]}\nTest accuracy: {test_scores[n]}\nTraining time: {traintime[n]}\nTraining epochs: {nns[n].training_epochs} \n\n')
            output_file.write('Test Data Confusion Matrix\n' + str(nns[n].confusion_matrix(x_test,y_test)) + '\n\n')
    elif part == 'c':
        nns = []
        traintime = []
        hidden_layer_units = [5,10,15,20,25]
        for n in tqdm.tqdm(hidden_layer_units):
            
            nnc = neuralnetwork(hidden_layers=[n],activation=sigmoid,activation_derivative=sigmoid_derivative,adaptive=True)
            start = time.time()
            nnc.fit(x_train,y_train)
            end = time.time()
            traintime.append(end-start)   
            nns.append(nnc)  
        train_scores = []
        test_scores = []
        for nn in nns:
            train_scores.append(nn.score(x_train,y_train))
            test_scores.append(nn.score(x_test,y_test))
        fig,ax = plt.subplots()
        ax.plot(hidden_layer_units,train_scores,label = 'train')
        ax.plot(hidden_layer_units,test_scores,label = 'test')
        ax.set_xlabel('Number of hidden units')
        ax.set_ylabel('Accuracy')
        ax.legend()
        plt.savefig(f'{output_folder}/{part}_accuracy.png')
        plt.close()
        fig,ax = plt.subplots()
        ax.plot(hidden_layer_units,traintime)
        ax.set_xlabel('Number of hidden units')
        ax.set_ylabel('Training time')
        plt.savefig(f'{output_folder}/{part}_training_time.png')
        plt.close()
        for n in range(len(hidden_layer_units)):
            output_file.write(f'Hidden units: {hidden_layer_units[n]}\nTrain accuracy: {train_scores[n]}\nTest accuracy: {test_scores[n]}\nTraining time: {traintime[n]}\nTraining epochs: {nns[n].training_epochs} \n\n')
            output_file.write('Test Data Confusion Matrix\n' + str(nns[n].confusion_matrix(x_test,y_test)) + '\n\n')
    elif part == 'd':
        nnd_sigmoid = neuralnetwork(hidden_layers = [100,100],activation = sigmoid,activation_derivative = sigmoid_derivative,adaptive=True)
        start = time.time()
        nnd_sigmoid.fit(x_train,y_train)
        end = time.time()
        output_file.write('Sigmoid activation function\n')
        output_file.write(f'Train accuracy: {nnd_sigmoid.score(x_train,y_train)}\nTest accuracy: {nnd_sigmoid.score(x_test,y_test)}\nTraining time: {end-start}\nTraining epochs: {nnd_sigmoid.training_epochs}\n\n')
        output_file.write('Test Data Confusion Matrix\n' + str(nnd_sigmoid.confusion_matrix(x_test,y_test)) + '\n\n')

        nnd_relu = neuralnetwork(hidden_layers = [100,100],activation = relu,activation_derivative = relu_derivative,adaptive=True)
        start = time.time()
        nnd_relu.fit(x_train,y_train)
        end = time.time()
        output_file.write('ReLU activation function\n')
        output_file.write(f'Train accuracy: {nnd_relu.score(x_train,y_train)}\nTest accuracy: {nnd_relu.score(x_test,y_test)}\nTraining time: {end-start}\nTraining epochs: {nnd_relu.training_epochs}\n\n')
        output_file.write('Test Data Confusion Matrix\n' + str(nnd_relu.confusion_matrix(x_test,y_test)) + '\n\n')
        
    elif part == 'e':
        nns = []
        traintime = []
        num_hidden_layer = [2,3,4,5]
        # Sigmoid activation function
        for n in tqdm.tqdm(num_hidden_layer):
            
            nne = neuralnetwork(hidden_layers=[50]*n,activation=sigmoid,activation_derivative=sigmoid_derivative,adaptive=True)
            start = time.time()
            nne.fit(x_train,y_train)
            end = time.time()
            traintime.append(end-start)   
            nns.append(nne)  
        train_scores = []
        test_scores = []
        for nn in nns:
            train_scores.append(nn.score(x_train,y_train))
            test_scores.append(nn.score(x_test,y_test))
        fig,ax = plt.subplots()
        ax.plot(num_hidden_layer,train_scores,label = 'train')
        ax.plot(num_hidden_layer,test_scores,label = 'test')
        ax.set_xlabel('Number of hidden layers')
        ax.set_ylabel('Accuracy')
        ax.legend()
        plt.savefig(f'{output_folder}/{part}_sigmoid_accuracy.png')
        plt.close()
        fig,ax = plt.subplots()
        ax.plot(num_hidden_layer,traintime)
        ax.set_xlabel('Number of hidden layers')
        ax.set_ylabel('Training time')
        plt.savefig(f'{output_folder}/{part}_sigmoid_training_time.png')
        plt.close()
        output_file.write('Sigmoid activation function\n')
        for n in range(len(num_hidden_layer)):
            output_file.write(f'Hidden Layer Count: {num_hidden_layer[n]}\nTrain accuracy: {train_scores[n]}\nTest accuracy: {test_scores[n]}\nTraining time: {traintime[n]}\nTraining epochs: {nns[n].training_epochs} \n\n')
            output_file.write('Test Data Confusion Matrix\n' + str(nns[n].confusion_matrix(x_test,y_test)) + '\n\n')
        
        # ReLU activation function
        nns = []
        traintime = []
        for n in tqdm.tqdm(num_hidden_layer):
                
                nne = neuralnetwork(hidden_layers=[50]*n,activation=relu,activation_derivative=relu_derivative,adaptive=True)
                start = time.time()
                nne.fit(x_train,y_train)
                end = time.time()
                traintime.append(end-start)   
                nns.append(nne)
        train_scores = []
        test_scores = []
        for nn in nns:
            train_scores.append(nn.score(x_train,y_train))
            test_scores.append(nn.score(x_test,y_test))
        fig,ax = plt.subplots()
        ax.plot(num_hidden_layer,train_scores,label = 'train')
        ax.plot(num_hidden_layer,test_scores,label = 'test')
        ax.set_xlabel('Number of hidden layers')
        ax.set_ylabel('Accuracy')
        ax.legend()
        plt.savefig(f'{output_folder}/{part}_relu_accuracy.png')
        plt.close()
        fig,ax = plt.subplots()
        ax.plot(num_hidden_layer,traintime)
        ax.set_xlabel('Number of hidden layers')
        ax.set_ylabel('Training time')
        plt.savefig(f'{output_folder}/{part}_relu_training_time.png')
        plt.close()
        output_file.write('ReLU activation function\n')
        for n in range(len(num_hidden_layer)):
            output_file.write(f'Hidden Layer Count: {num_hidden_layer[n]}\nTrain accuracy: {train_scores[n]}\nTest accuracy: {test_scores[n]}\nTraining time: {traintime[n]}\nTraining epochs: {nns[n].training_epochs} \n\n')
            output_file.write('Test Data Confusion Matrix\n' + str(nns[n].confusion_matrix(x_test,y_test)) + '\n\n')

    elif part == 'f':
        nnf = neuralnetwork(hidden_layers = [50]*2,activation = relu,activation_derivative = relu_derivative,adaptive=True,objective_function='BCE')
        start = time.time()
        nnf.fit(x_train,y_train)
        end = time.time()
        output_file.write(f'ReLU activation function , Layers : {str([50]*2)}, objective function : BCE\n')
        output_file.write(f'Train accuracy: {nnf.score(x_train,y_train)}\nTest accuracy: {nnf.score(x_test,y_test)}\nTraining time: {end-start}\nTraining epochs: {nnf.training_epochs}\n\n')
        output_file.write('Test Data Confusion Matrix\n' + str(nnf.confusion_matrix(x_test,y_test)) + '\n\n')
    elif part == 'g':
        
        clf = MLPClassifier(hidden_layer_sizes=(100,100),activation='relu',solver='sgd',max_iter=1000)
        start = time.time()
        clf.fit(x_train,y_train)
        end = time.time()
        output_file.write('Scikit-learn MLPClassifier\n')
        output_file.write(f'Train accuracy: {clf.score(x_train,y_train)}\nTest accuracy: {clf.score(x_test,y_test)}\nTraining time: {end-start}\n')
        output_file.write('Test Data Confusion Matrix\n' + str(confusion_matrix(y_test,clf.predict(x_test))) + '\n\n')
if __name__ == '__main__':
    main()

