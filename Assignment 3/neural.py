from MyNN import one_hot_encoder,sigmoid,sigmoid_derivative,relu,relu_derivative, neuralnetwork
import numpy as np
import scipy
import pandas as pd
import tqdm
import sys
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
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
            
            nnb = neuralnetwork(layers=[n],activation=sigmoid,activation_derivative=sigmoid_derivative)
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
            
            nnc = neuralnetwork(layers=[n],activation=sigmoid,activation_derivative=sigmoid_derivative,adaptive=True)
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
        nnd_sigmoid = neuralnetwork(layers = [100,100],activation = sigmoid,activation_derivative = sigmoid_derivative,adaptive=True)
        start = time.time()
        nnd_sigmoid.fit(x_train,y_train)
        end = time.time()
        output_file.write('Sigmoid activation function\n')
        output_file.write(f'Train accuracy: {nnd_sigmoid.score(x_train,y_train)}\nTest accuracy: {nnd_sigmoid.score(x_test,y_test)}\nTraining time: {end-start}\nTraining epochs: {nnd_sigmoid.training_epochs}\n\n')
        output_file.write('Test Data Confusion Matrix\n' + str(nnd_sigmoid.confusion_matrix(x_test,y_test)) + '\n\n')

        nnd_relu = neuralnetwork(layers = [100,100],activation = relu,activation_derivative = relu_derivative,adaptive=True)
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
            
            nne = neuralnetwork(layers=[50]*n,activation=sigmoid,activation_derivative=sigmoid_derivative,adaptive=True)
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
                
                nne = neuralnetwork(layers=[50]*n,activation=relu,activation_derivative=relu_derivative,adaptive=True)
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
        nnf = neuralnetwork(layers = [50]*2,activation = relu,activation_derivative = relu_derivative,adaptive=True,objective_function='BCE')
        start = time.time()
        nnf.fit(x_train,y_train)
        end = time.time()
        output_file.write(f'ReLU activation function , Layers : {str([50]*2)}, objective function : BCE\n')
        output_file.write(f'Train accuracy: {nnf.score(x_train,y_train)}\nTest accuracy: {nnf.score(x_test,y_test)}\nTraining time: {end-start}\nTraining epochs: {nnf.training_epochs}\n\n')
    elif part == 'g':
        start = time.time()
        clf = MLPClassifier(hidden_layer_sizes=(100,100),activation='relu',solver='sgd',max_iter=1000)
        end = time.time()
        clf.fit(x_train,y_train)
        output_file.write('Scikit-learn MLPClassifier\n')
        output_file.write(f'Train accuracy: {clf.score(x_train,y_train)}\nTest accuracy: {clf.score(x_test,y_test)}\nTraining time: {end-start}\n')

if __name__ == '__main__':
    main()

