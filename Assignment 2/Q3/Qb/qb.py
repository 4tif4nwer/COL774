import sys
import numpy as np
import pandas as pd
from cvxopt import matrix
import random
import matplotlib.pyplot as plt
from scipy import stats as st
import time
from sklearn.svm import SVC

def confusion_matrix(Y_test, Y_pred, k):
    cmat = np.zeros((k, k))
    for i, j in zip(Y_pred, Y_test):
        cmat[int(i) - 1][int(j) - 1] += 1
    return cmat


def data_loader(train_data,test_data):
    obj = pd.read_pickle(f"{train_data}/train_data.pickle")

    X_full_train = np.reshape(obj['data'],(len(obj['data']),32*32*3))
    X_full_train = np.array(X_full_train,dtype = 'float64')
    Y_full_train = np.reshape(obj['labels'],(X_full_train.shape[0],1))
    X_full_train/=255

    obj = pd.read_pickle(r'D:/GitHub/COL774/Assignment 2/data/part2_data/test_data.pickle')

    X_test = np.reshape(obj['data'],(len(obj['data']),32*32*3))
    X_test = np.array(X_test,dtype = 'float64')
    Y_test = np.reshape(obj['labels'],(len(obj['labels']),1))
    X_test/=255

    return X_full_train,Y_full_train,X_test,Y_test

def main():
    
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    print("Multi-Class Classification by SVM(ovo) using sklearn")

    X_full_train,Y_full_train,X_test,Y_test = data_loader(train_data,test_data)

    
    clf =SVC(C = 1.0, kernel = 'rbf', gamma=0.001,decision_function_shape='ovo')
    
    start = time.time()
    clf.fit(X_full_train,Y_full_train.reshape(-1,))
    end = time.time()
    
    Predictions = (clf.predict(X_test))
    acc = len(np.where(Predictions == Y_test.reshape(-1,))[0])/len(Y_test)
    
    print(f"Test Accuracy : {100*acc}%")
    print(f"Training time : {end-start}")
    cmat_sklearn = confusion_matrix(Y_test.reshape(-1,),Predictions,5)
    np.save("cmat_sklearn.npy",cmat_sklearn)
    
    # Misclassified Objects
    misobj = []
    for i in range(len(Predictions)):
        if Predictions[i] != Y_test[i,0]:
            misobj.append(i)
    
    random.shuffle(misobj)
    i = 1
    for img in misobj:
        plt.imshow(X_test[img].reshape((32,32,3)))
        plt.savefig(f"sklearn_misclf{i}_{int(Predictions[img])}.png")
        plt.savefig(f"sklearn_misclf{i}.png")
        
        plt.close()
        i += 1
        if i == 11:
            break

    

if __name__ == '__main__':
    main()
    