import sys
import numpy as np
import pandas as pd
import cvxopt as co
from cvxopt import matrix
import time
import os
from matplotlib import pyplot as plt
from sklearn.svm import SVC

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

    
    # We wil classify 0 and 4 since my entry number ends with 9 .
    class1 = 0
    class2 = 4

    X_train,Y_train,X_test,Y_test = data_loader(train_data,test_data,class1,class2)

    # Linear

    print("Linear Kernel SVM with sklearn")

    
    clf =SVC(C = 1.0, kernel = 'linear', gamma=0.001)
    start = time.time()

    clf.fit(X_train,Y_train.reshape(-1,))

    end = time.time()
    Y_pred = (clf.predict(X_test))
    acc = len(np.where(Y_pred == Y_test.reshape(-1,))[0])/len(Y_test)
    print(f"No. of Support Vectors = {len(clf.support_)}, which constitutes {100*len(clf.support_)/Y_train.shape[0]}% of training samples")
    try :
        os.chdir("..")
        supportVectors =  np.load("Qa/supportVectors.npy")
        print(f"Common Support Vectors = {len(np.intersect1d(supportVectors.reshape(-1,),np.array(clf.support_)))}")
        os.chdir("./Qc")
    except :
        print("Common Support Vectors can't be determined")
        
    print(f"Test Accuracy : {100*acc}%")  
    print(f"Computation time = {end-start}") 
    alpha = clf.dual_coef_[0]
    alpha = alpha.reshape((-1,1))
    w = np.sum(alpha*X_train[clf.support_],axis = 0)

    m = X_train.shape[0]
    inf = float('inf')
    negsupport, possupport = -inf, inf
    
    for i in range(m):

        val = np.dot(X_train[i],w)
        
        if int(Y_train[i]) == -1:
            negsupport = max(negsupport, val)
        else:
            possupport = min(possupport, val)

    b = -(negsupport + possupport) / 2

    print(f"w = {w}\nb = {b}")

    print("Gaussian Kernel SVM with sklearn")

    clf = SVC(C = 1.0,kernel = 'rbf', gamma=0.001)
    start = time.time()
    clf.fit(X_train,Y_train.reshape(-1,))
    end = time.time()
    Y_pred = (clf.predict(X_test))
    acc = len(np.where(Y_pred == Y_test.reshape(-1,))[0])/len(Y_test)
    
    print(f"No. of Support Vectors = {len(clf.support_)}, which constitutes {100*len(clf.support_)/Y_train.shape[0]}% of training samples")

    try :
        os.chdir("..")
        supportVectors =  np.load("Qb/supportVectors.npy")
        print(f"Common Support Vectors = {len(np.intersect1d(supportVectors.reshape(-1,),np.array(clf.support_)))}")
        os.chdir("./Qc")
    except :
        print("Common Support Vectors can't be determined")
        
    

    print(f"Test Accuracy : {100*acc}%")   
    print(f"Computation time = {end-start}")

if __name__ == '__main__':
    main()
    