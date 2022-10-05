import sys
import numpy as np
import pandas as pd
import cvxopt as co
from cvxopt import matrix
import time
import time
from matplotlib import pyplot as plt

def LinearKernelSVM(x,y,C = 1.0):    

    Z = x*y
    m = x.shape[0]
    P = matrix(np.dot(Z,Z.T))
    
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
    
    alpha = sol['x']
    
    w = np.sum(alpha*Z,axis = 0)
    
    eps = 1e-5
    supportVectors = [i for i in range(m) if abs(alpha[i]) > eps]
    
    inf = float('inf')
    negsupport, possupport = -inf, inf
    
    for i in range(m):

        val = np.dot(x[i],w)
        
        if int(y[i]) == -1:
            negsupport = max(negsupport, val)
        else:
            possupport = min(possupport, val)

    b = -(negsupport + possupport) / 2

    return w,b,supportVectors,np.array(alpha).T

def data_loader(train_data,test_data,class1,class2):
    
    obj = pd.read_pickle(f"{train_data}/train_data.pickle")

    X_full = np.reshape(obj['data'],(len(obj['data']),32*32*3))
    Y_full = np.reshape(obj['labels'],(len(obj['labels']),1))

    X_train = np.zeros((len(np.where(Y_full==class1)[0])+len(np.where(Y_full==class2)[0]),32*32*3))
    Y_train = np.zeros((len(np.where(Y_full==class1)[0])+len(np.where(Y_full==class2)[0]),1))

    X_train[0:len(np.where(Y_full==class1)[0])] = X_full[np.where(Y_full==class1)[0]]
    X_train[len(np.where(Y_full==class1)[0]):] = X_full[np.where(Y_full==class2)[0]]
    X_train/=255

    Y_train[0:len(np.where(Y_full==class1)[0]),0] = 1
    Y_train[len(np.where(Y_full==class1)[0]):,0] = -1

    obj = pd.read_pickle(f"{test_data}/test_data.pickle")

    X_full = np.reshape(obj['data'],(len(obj['data']),32*32*3))
    Y_full = np.reshape(obj['labels'],(len(obj['labels']),1))
    

    X_test = np.zeros((len(np.where(Y_full==class1)[0])+len(np.where(Y_full==class2)[0]),32*32*3))
    Y_test = np.zeros((len(np.where(Y_full==class1)[0])+len(np.where(Y_full==class2)[0]),1))

    X_test[0:len(np.where(Y_full==class1)[0])] = X_full[np.where(Y_full==class1)[0]]
    X_test[len(np.where(Y_full==class1)[0]):] = X_full[np.where(Y_full==class2)[0]]
    X_test /= 255

    Y_test[0:len(np.where(Y_full==class1)[0]),0] = 1
    Y_test[len(np.where(Y_full==class1)[0]):,0] = -1

    return X_train,Y_train,X_test,Y_test

def main():

    train_data = sys.argv[1]
    test_data = sys.argv[2]

    print("Linear Kernel SVM using CVXOPT")

    # We wil classify 0 and 4 since my entry number ends with 9 .
    class1 = 0
    class2 = 4

    X_train,Y_train,X_test,Y_test = data_loader(train_data,test_data,class1,class2)

    # Train

    start = time.time()

    w,b,supportVectors,alpha = LinearKernelSVM(X_train,Y_train)

    end = time.time()


    print(f"No. of Support Vectors = {len(supportVectors)}, which constitutes {100*len(supportVectors)/Y_train.shape[0]}% of training samples")
    np.save("supportVectors.npy",np.array(supportVectors))
    print(f"w = {w}\nb = {b}")
    print(f"Computation time = {end-start}")

    # Test

    Y_pred = (np.dot(X_test,w)+b).reshape((Y_test.shape[0],1))
    Y_pred[np.where(Y_pred>0)[0]] = 1
    Y_pred[np.where(Y_pred<=0)[0]] = -1
    acc = len(np.where(Y_pred == Y_test)[0])/len(Y_pred)

    print(f"Test Accuracy : {100*acc}%")

    top5 = X_train[np.argsort(alpha[0])[-5:]]
    top5 = top5.reshape(5,32,32,3)

    plt.rcParams['figure.dpi'] = 70
    plt.rcParams['figure.figsize']=[10,10]   
    for i in range(5):
        plt.imshow(top5[i])
        plt.savefig(f"LinearTop5_SV{i}.png")
        plt.close()

if __name__ == '__main__':
    main()
    