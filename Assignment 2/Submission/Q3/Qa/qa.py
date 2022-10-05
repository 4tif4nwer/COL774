import sys
import numpy as np
import pandas as pd
import cvxopt as co
from cvxopt import matrix
from scipy import stats as st
import matplotlib.pyplot as plt
import time
import random

def EuclideanDistanceMatrix(x, y):
    squares_x = np.einsum('ij,ij->i', x, x)
    squares_y = np.einsum('ij,ij->i', y, y)
    return np.tile(squares_y, (squares_x.shape[0], 1)) + np.tile(squares_x, (squares_y.shape[0], 1)).T - 2 * np.matmul(x, y.T)

def confusion_matrix(Y_test, Y_pred, classes):
    cmat = np.zeros((classes, classes))
    for i, j in zip(Y_pred, Y_test):
        cmat[int(i) - 1][int(j) - 1] += 1
    return cmat

def GaussianKernelSVM(x,y,C = 1.0,gamma = 0.001):    
    
    m = len(y)
    squares = np.exp(-gamma * EuclideanDistanceMatrix(x, x))
    
    P = matrix(squares * np.outer(y, y))

    q = -np.ones((m,1))
    q = matrix(q)
    
    G = np.identity(m)
    G = np.append(G, -G, axis=0)
    G = matrix(G)
    
    h = np.ones((m,1))
    h = matrix(np.append(C * h, 0 * h, axis=0))
    
    A = matrix([[y[i,0]] for i in range(m)])
    b = matrix(0.0)
    
    sol = co.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    
    alpha = (np.array(sol['x']).T)[0]
    
    eps = 1e-5
    supportVectors = [i for i in range(m) if abs(alpha[i]) > eps]
    
    value = np.einsum('i,i,ij->j', alpha, y.reshape((m,)), squares)

    inf = float('inf')
    negsupport, possupport = -inf, inf
   
    for i in range(m):
        val = value[i]
        
        if int(y[i]) == -1:
            negsupport = max(negsupport, val)
        else:
            possupport = min(possupport, val)

    b = -(negsupport+possupport) / 2

    return b,supportVectors,alpha

def testGaussian(x,x_sv,y_sv,gamma,alpha,b):
    nsv = x_sv.shape[0]
    m = x.shape[0]
    E = EuclideanDistanceMatrix(x_sv,x)
    Y_pred = np.einsum('i,ij->j', (alpha * y_sv).reshape(nsv,), np.exp(-gamma * E)) + b
    return Y_pred.T


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

def SVM_train(X_full_train,Y_full_train):
    Classifiers = dict()

    for i in range(5):
        for j in range(i+1,5):
            
            class1 = i
            class2 = j

            X_train = np.zeros((len(np.where(Y_full_train==class1)[0])+len(np.where(Y_full_train==class2)[0]),32*32*3))
            Y_train = np.zeros((len(np.where(Y_full_train==class1)[0])+len(np.where(Y_full_train==class2)[0]),1))

            X_train[0:len(np.where(Y_full_train==class1)[0])] = X_full_train[np.where(Y_full_train==class1)[0]]
            X_train[len(np.where(Y_full_train==class1)[0]):] = X_full_train[np.where(Y_full_train==class2)[0]]
            

            Y_train[0:len(np.where(Y_full_train==class1)[0])] = 1
            Y_train[len(np.where(Y_full_train==class1)[0]):] = -1
            
            b,supportVectors,alpha = GaussianKernelSVM(X_train,Y_train)
            Classifiers[(i,j)] = (b, supportVectors,alpha)
    
    return Classifiers

def SVM_classify(Classifiers,X_test,X_full_train,Y_full_train):
    Y_pred = np.zeros((X_test.shape[0],10))

    binary_cfr_iter = 0


    for i in range(5):
        for j in range(i+1,5):
    
            class1 = i
            class2 = j
            b,supportVectors,alpha = Classifiers[(class1,class2)]
            X_train = np.zeros((len(np.where(Y_full_train==class1)[0])+len(np.where(Y_full_train==class2)[0]),32*32*3))
            Y_train = np.zeros((len(np.where(Y_full_train==class1)[0])+len(np.where(Y_full_train==class2)[0]),1))

            X_train[0:len(np.where(Y_full_train==class1)[0])] = X_full_train[np.where(Y_full_train==class1)[0]]
            X_train[len(np.where(Y_full_train==class1)[0]):] = X_full_train[np.where(Y_full_train==class2)[0]]
            
            Y_train[0:len(np.where(Y_full_train==class1)[0])] = 1
            Y_train[len(np.where(Y_full_train==class1)[0]):] = -1
            
            Y_pred[:,binary_cfr_iter] = testGaussian(X_test,X_train[supportVectors],Y_train[supportVectors],0.001,alpha[supportVectors].reshape(-1,1),b)
            Y_pred[np.where(Y_pred[:,binary_cfr_iter]>=0)[0],binary_cfr_iter] = class1
            Y_pred[np.where(Y_pred[:,binary_cfr_iter]<0)[0],binary_cfr_iter] = class2
            binary_cfr_iter += 1

    m = X_test.shape[0]

    Predictions = np.zeros(m)
    
    for i in range(m):
        Predictions[i] = st.mode(Y_pred[i])[0]

    return Predictions

def main():

    train_data = sys.argv[1]
    test_data = sys.argv[2]

    print("Multi-Class Classification by SVM(ovo) using CVXOPT")

    X_full_train,Y_full_train,X_test,Y_test = data_loader(train_data,test_data)

    start = time.time()

    Classifiers = SVM_train(X_full_train,Y_full_train)

    end = time.time()

    Predictions = SVM_classify(Classifiers,X_test,X_full_train,Y_full_train)

    acc = len(np.where(Predictions == Y_test.reshape(-1,))[0])/len(Y_test)

    print(f"Test Accuracy : {100*acc}%")
    print(f"Training time : {end-start}")

    cmat_cvxopt = confusion_matrix(Y_test.reshape(-1,),Predictions,5)
    np.save("cmat_cvxopt.npy",cmat_cvxopt)
    
    # Misclassified Objects
    misobj = []
    for i in range(len(Predictions)):
        if Predictions[i] != Y_test[i,0]:
            misobj.append(i)
    
    random.shuffle(misobj)
    i = 1
    for img in misobj:
        plt.imshow(X_test[img].reshape((32,32,3)))
        plt.savefig(f"cvxopt_misclf{i}.png")
        
        plt.close()
        i += 1
        if i == 11:
            break

if __name__ == '__main__':
    main()
    