from cProfile import label
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import pinv
import tqdm

def sigmoid(parameters,x):
    return 1/(1+np.exp(-np.transpose(np.dot(parameters,np.transpose(x)))))

def mean_normalization(x):
    return (x-np.mean(x,axis=0))/np.std(x,axis=0),np.mean(x,axis=0),np.std(x,axis=0)

def prediction(parameters,x):
    return sigmoid(parameters,x)

def gradient_descent(parameters, y, x):
    hypothesis = prediction(parameters,x)
    hessian = -np.dot((hypothesis*x).T,(1-hypothesis)*x)
    Error = y - hypothesis
    DelLL = np.sum(Error*x,axis = 0).T
    return -np.dot(pinv(hessian),DelLL)

logisticX = pd.read_csv("logisticX.csv")
logisticY = pd.read_csv("logisticY.csv")

x_data = logisticX.to_numpy()
x,norm_mean, norm_std = mean_normalization(x_data)
x_full = np.ones((x.shape[0],x.shape[1]+1))
x_full[:,1:] = x
x = x_full
y = y_data = logisticY.to_numpy()
# print(x,y)

parameters = np.zeros((1,x.shape[1]))
for i in tqdm.tqdm(range(1000)):
    change = gradient_descent(parameters,y,x)
    parameters = parameters + change
print(parameters)

plt.rcParams['figure.dpi'] = 70
plt.rcParams['figure.figsize']=[12,12]
fig,ax = plt.subplots()
ax.scatter(x_data[np.where(y==1),0],x_data[np.where(y==1),1],marker='x',label = "Category 1")
ax.scatter(x_data[np.where(y==0),0],x_data[np.where(y==0),1],marker='s',label = "Category 0")
x_axis = np.linspace(2,7,10)
x_norm = (x_axis-norm_mean[0])/norm_std[0]
y_norm = ((0.5-parameters[0,0])-parameters[0,1]*x_norm)/parameters[0,2]
y_axis = y_norm*norm_std[1] + norm_mean[1]
ax.plot(x_axis,y_axis,c = 'red', label = "Decision Boundary")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
plt.show()
