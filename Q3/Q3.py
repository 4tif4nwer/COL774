from cProfile import label
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import pinv,norm
import tqdm
import sys

def sigmoid(z):
    return 1/(1+np.exp(-z))

def mean_normalization(x):
    return (x-np.mean(x,axis=1).reshape(x.shape[0],1))/np.std(x,axis=1).reshape(x.shape[0],1),np.mean(x,axis=1).reshape(x.shape[0],1),np.std(x,axis=1).reshape(x.shape[0],1)

def prediction(parameters,x):
    return sigmoid((np.dot(parameters.T,x)))

def gradient_descent(parameters, y, x):
    hypothesis = prediction(parameters,x).reshape((1,x.shape[1]))
    hessian = -np.dot((hypothesis*x),((1-hypothesis)*x).T).reshape((parameters.shape[0],parameters.shape[0]))
    Error = (y - hypothesis).reshape((1,x.shape[1]))
    DelLL = np.sum(Error*x,axis = 1).reshape(parameters.shape)
    return -np.dot(pinv(hessian),DelLL).reshape(parameters.shape)

def main():

    train_data_directory = sys.argv[1]
    test_data_directory = sys.argv[2]
    logisticX = pd.read_csv(f"{train_data_directory}/X.csv",header=None)
    logisticY = pd.read_csv(f"{train_data_directory}/Y.csv",header = None)

    x_data = logisticX.to_numpy()
    x_data = x_data.T
    x,norm_mean, norm_std = mean_normalization(x_data)
    x_full = np.ones((x.shape[0]+1,x.shape[1]))
    x_full[1:,:] = x[:]
    x = x_full

    y = y_data = logisticY.to_numpy()
    y = y.T

    
    min_change = 1e-7

    parameters = np.zeros((x.shape[0],1))
    while True:
        change = gradient_descent(parameters,y,x)
        parameters = parameters + change.reshape(parameters.shape)
        if norm(change)<min_change:
            break
    print(parameters)

    plt.rcParams['figure.dpi'] = 70
    plt.rcParams['figure.figsize']=[12,12]
    fig,ax = plt.subplots()
    ax.scatter(x_data[0,np.where(y==1)[1]],x_data[1,np.where(y==1)[1]],marker='x',label = "Category 1")
    ax.scatter(x_data[0,np.where(y==0)[1]],x_data[1,np.where(y==0)[1]],marker='s',label = "Category 0")
    x_axis = np.linspace(2,7,10)
    x_norm = (x_axis-norm_mean[0])/norm_std[0]
    y_norm = ((0.5-parameters[0,0])-parameters[1,0]*x_norm)/parameters[2,0]
    y_axis = y_norm*norm_std[1] + norm_mean[1]
    ax.plot(x_axis,y_axis,c = 'red', label = "Decision Boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Logistic Regression Decision Boundary")
    ax.legend()

    fig.savefig("Decision_Boundary.png")
    plt.close()

    # Test Data
    logisticX = pd.read_csv(f"{test_data_directory}/X.csv",header=None)

    x_data = logisticX.to_numpy()
    x_data = x_data.T
    x,norm_mean, norm_std = mean_normalization(x_data)
    x_full = np.ones((x.shape[0]+1,x.shape[1]))
    x_full[1:,:] = x[:]
    x = x_full
    y = prediction(parameters,x)

    y[np.where(y>0.5)] = 1
    y[np.where(y<=0.5)] = 0
    np.savetxt("result_3.txt",y,fmt='%f')


if __name__ == '__main__':
    main()
    