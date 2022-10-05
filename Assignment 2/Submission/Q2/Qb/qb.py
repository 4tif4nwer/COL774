import sys
import numpy as np
import pandas as pd
import time
import cvxopt as co
from cvxopt import matrix
import time
from matplotlib import pyplot as plt

def EuclideanDistanceMatrix(x, y):
    squares_x = np.einsum('ij,ij->i', x, x)
    squares_y = np.einsum('ij,ij->i', y, y)
    return np.tile(squares_y, (squares_x.shape[0], 1)) + np.tile(squares_x, (squares_y.shape[0], 1)).T - 2 * np.matmul(x, y.T)

def GaussianKernelSVM(x,y,C = 1.0,gamma = 0.001):    
    
    m = len(y)
    Kernel = np.exp(-gamma * EuclideanDistanceMatrix(x, x))
    
    P = matrix(Kernel * np.outer(y, y))

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
    
    value = np.einsum('i,i,ij->j', alpha, y.reshape((m,)), Kernel)

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

def data_loader(train_data,test_data,class1,class2):
    
    obj = pd.read_pickle(f"{train_data}/train_data.pickle")

    X_full = np.reshape(obj['data'],(len(obj['data']),32*32*3))
    Y_full = np.reshape(obj['labels'],(len(obj['labels']),1))

    X_train = np.zeros((len(np.where(Y_full==class1)[0])+len(np.where(Y_full==class2)[0]),32*32*3))
    Y_train = np.zeros((len(np.where(Y_full==class1)[0])+len(np.where(Y_full==class2)[0]),1))

    X_train[0:len(np.where(Y_full==class1)[0])] = X_full[np.where(Y_full==class1)[0]]
    X_train[len(np.where(Y_full==class1)[0]):] = X_full[np.where(Y_full==class2)[0]]
    X_train/=255

    Y_train[0:len(np.where(Y_full==class1)[0])] = 1
    Y_train[len(np.where(Y_full==class1)[0]):] = -1

    obj = pd.read_pickle(f"{test_data}/test_data.pickle")

    X_full = np.reshape(obj['data'],(len(obj['data']),32*32*3))
    Y_full = np.reshape(obj['labels'],(len(obj['labels']),1))
    

    X_test = np.zeros((len(np.where(Y_full==class1)[0])+len(np.where(Y_full==class2)[0]),32*32*3))
    Y_test = np.zeros((len(np.where(Y_full==class1)[0])+len(np.where(Y_full==class2)[0])))

    X_test[0:len(np.where(Y_full==class1)[0])] = X_full[np.where(Y_full==class1)[0]]
    X_test[len(np.where(Y_full==class1)[0]):] = X_full[np.where(Y_full==class2)[0]]
    X_test /= 255

    Y_test[0:len(np.where(Y_full==class1)[0])] = 1
    Y_test[len(np.where(Y_full==class1)[0]):] = -1

    return X_train,Y_train,X_test,Y_test


def main():

    train_data = sys.argv[1]
    test_data = sys.argv[2]

    obj = pd.read_pickle(f"{train_data}/train_data.pickle")

    # We wil classify 0 and 4 since my entry number ends with 9 .
    class1 = 0
    class2 = 4

    print("Gaussian Kernel SVM using CVXOPT")

    X_train,Y_train,X_test,Y_test = data_loader(train_data,test_data,class1,class2)

    start = time.time()
    b,supportVectors,alpha = GaussianKernelSVM(X_train,Y_train)
    end = time.time()

    print(f"No. of Support Vectors = {len(supportVectors)}, which constitutes {100*len(supportVectors)/Y_train.shape[0]}% of training samples")
    np.save("supportVectors.npy",np.array(supportVectors))
    print(f"Computation time = {end-start}")
    # Test

    Y_pred = testGaussian(X_test,X_train[supportVectors],Y_train[supportVectors],0.001,alpha[supportVectors].reshape(-1,1),b)

    Y_pred[np.where(Y_pred>=0)[0]] = 1
    Y_pred[np.where(Y_pred<0)[0]] = -1

    acc = len(np.where(Y_pred == Y_test)[0])/len(Y_test)

    print(f"Test Accuracy : {100*acc}%")

    top5 = X_train[np.argsort(alpha)[-5:]]
    top5 = top5.reshape(5,32,32,3)

    plt.rcParams['figure.dpi'] = 70
    plt.rcParams['figure.figsize']=[10,10]   
    for i in range(5):
        plt.imshow(top5[i])
        plt.savefig(f"GaussianTop5_SV{i}.png")
        plt.close()

if __name__ == '__main__':
    main()
    