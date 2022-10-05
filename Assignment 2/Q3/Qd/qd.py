from random import shuffle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def data_loader(train_data,test_data):
    obj = pd.read_pickle(f"{train_data}/train_data.pickle")

    X_full_train = np.reshape(obj['data'],(len(obj['data']),32*32*3))
    X_full_train = np.array(X_full_train,dtype = 'float64')
    Y_full_train = np.reshape(obj['labels'],(X_full_train.shape[0],1))
    X_full_train/=255

    obj = pd.read_pickle(r'D:/GitHub/COL774/Assignment 2/data/part2_data/test_data.pickle')

    X_full_test = np.reshape(obj['data'],(len(obj['data']),32*32*3))
    X_full_test = np.array(X_full_test,dtype = 'float64')
    Y_full_test = np.reshape(obj['labels'],(len(obj['labels']),1))
    X_full_test/=255

    return X_full_train,Y_full_train,X_full_test,Y_full_test



def kfoldcrossval(training_data,C,folds = 5):
    clf =SVC(C = C, kernel = 'rbf', gamma=0.001,decision_function_shape='ovo')
    valacc = 0

    for train_index, test_index in training_data:
        
        X_train, X_test = X_full_train[train_index], X_full_train[test_index]
        Y_train, Y_test = Y_full_train[train_index].reshape(-1,), Y_full_train[test_index].reshape(-1,)
        clf.fit(X_train,Y_train.reshape(-1,))
        Predictions = (clf.predict(X_test))
        valacc += len(np.where(Predictions == Y_test.reshape(-1,))[0])/len(Y_test)

    valacc = valacc/folds
    print(f"Val Accuracy : {valacc}")
    
    clf.fit(X_full_train,Y_full_train.reshape(-1,))
    Predictions = (clf.predict(X_full_test))
    testacc = len(np.where(Predictions == Y_full_test.reshape(-1,))[0])/len(Y_full_test)
    print(f"Test Accuracy : {testacc}")
    return valacc,testacc 
    

def main():
    
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    print("Multi-Class Classification with Cross-Validation by SVM(ovo) using sklearn ")
    
    global X_full_train,Y_full_train,X_full_test,Y_full_test

    X_full_train,Y_full_train,X_full_test,Y_full_test = data_loader(train_data,test_data)

    folds = 5

    kf = KFold(n_splits=folds,shuffle = True)

    ValAcc = np.zeros(8)
    TestAcc = np.zeros(8)
    

    for C,i in zip([1e-5,1e-3,1,5,10,20,100,1000],range(8)):
        print(f"C = {C}")
        training_data = kf.split(X_full_train)
        ValAcc[i],TestAcc[i] = kfoldcrossval(training_data,C,folds)

    ValAcc = ValAcc * 100
    TestAcc = TestAcc * 100
    
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize']=[10,10]
    plt.rcParams.update({'font.size': 10})
    fig,ax = plt.subplots()
    plt.xscale('log')
    ax.plot([1e-5,1e-3,1,5,10,20,100,1000],ValAcc,c = 'blue',label = "Validation Accuracy")
    ax.plot([1e-5,1e-3,1,5,10,20,100,1000],TestAcc,c = 'red',label = "Test Accuracy")
    ax.set_title("Test and Validation Accuracies vs. C",fontsize=15)
    ax.set_xlabel("Regularization Parameter(C)",fontsize = 11)
    ax.set_ylabel("Accuracy %",fontsize = 11)
    ax.legend()
    plt.savefig("Accplot.png")
    plt.show()

    

if __name__ == '__main__':
    main()
    