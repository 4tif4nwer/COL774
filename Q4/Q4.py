from cProfile import label
import numpy as np
import scipy
from scipy.linalg import  pinv,det
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from cmath import log
from matplotlib import cm
import sys,os

def mean_normalization(x):
    return (x-np.mean(x,axis=1).reshape(x.shape[0],1))/np.std(x,axis=1).reshape(x.shape[0],1),np.mean(x,axis=1).reshape(x.shape[0],1),np.std(x,axis=1).reshape(x.shape[0],1)

def main():
    train_data_directory = sys.argv[1]
    test_data_directory = sys.argv[2]

    # datax = np.genfromtxt('q4x.dat',encoding=None,dtype=None)
    # datax = [list(item) for item in datax]
    # datax = np.array(datax,dtype = 'float64')
    # x = np.zeros(datax.shape)
    # x[:] = datax[:]
    # x = x.T
    # datax = datax.T
    # datax,norm_mean,norm_std = mean_normalization(datax)
    # # print(datax.shape)

    # datay = np.genfromtxt('q4y.dat',encoding = None,names = '',dtype=None)
    # datay = [list(item) for item in datay]
    # datay = np.array(datay)
    # y = np.zeros(datay.shape)
    # y[(np.where(datay=='Alaska'))[0]] = 1
    # datay = datay.T
    # y = y.T
    logisticX = pd.read_csv(f"{train_data_directory}/X.csv",header=None)
    logisticY = pd.read_csv(f"{train_data_directory}/Y.csv",header = None)

    datax = logisticX.to_numpy()
    datax = datax.T
    x = np.zeros_like(datax)
    x[:] = datax[:]
    datax,norm_mean, norm_std = mean_normalization(datax)

    datay = logisticY.to_numpy()
    y = np.zeros((datay.shape[0],1))
    y[(np.where(datay=='Alaska'))[0]] = 1
    y = y.T
    datay = datay.T

    # Linear Boundary Approximation

    phi = len((np.where(datay=='Alaska'))[0])/len(datay[0])
    print(f"phi : {phi}")

    mu0 = np.sum(datax[:,(np.where(datay=='Canada'))[1]],axis = 1).reshape((datax.shape[0],1)) / len((np.where(datay=='Canada'))[1])
    print(f"mu_0 : {mu0}")

    mu1 = np.sum(datax[:,(np.where(datay=='Alaska'))[1]],axis = 1).reshape((datax.shape[0],1)) / len((np.where(datay=='Alaska'))[1])
    print(f"mu_1 : {mu1}")

    diff = datax
    diff[:,(np.where(datay=='Canada'))[1]] = diff[:,(np.where(datay=='Canada'))[1]] - mu0
    diff[:,(np.where(datay=='Alaska'))[1]] = diff[:,(np.where(datay=='Alaska'))[1]] - mu1
    # print(diff)

    sigma = np.dot(diff,diff.T) / diff.shape[1]

    print(f"Sigma : {sigma}")

    sigmainv = pinv(sigma)

    intercept =  0.5*(np.dot(mu0.T,np.dot(sigmainv,mu0))-np.dot(mu1.T,np.dot(sigmainv,mu1))) + np.real((log(phi/(1-phi))))

    coeff1 = np.dot(sigmainv,(mu0-mu1))/2
    coeff2 = np.dot((mu0-mu1).T,sigmainv)/2

    coeff = coeff1 + coeff2.T
    plt.rcParams['figure.dpi'] = 80
    plt.rcParams['figure.figsize']=[15,15]
    fig,ax = plt.subplots()
    
    ax.scatter(x[0,(np.where(y==1))[1]],x[1,(np.where(y==1))[1]],c = 'red',marker = 'x',label  = 'Alaska')
    ax.scatter(x[0,(np.where(y==0))[1]],x[1,(np.where(y==0))[1]],c = 'blue',label = 'Canada')

    x_axis = np.linspace(60,160,10)
    x_norm = (x_axis-norm_mean[0])/norm_std[0]
    y_norm = (intercept - coeff[0,0]*x_norm)/coeff[1,0]
    y_axis = y_norm*norm_std[1] + norm_mean[1]

    ax.plot(x_axis,y_axis.T,c = "green", label = "Linear Decision Boundary")
    ax.legend()
    ax.set_xlabel("Fresh water growth ring diameter")
    ax.set_ylabel("Marine water growth ring diameter")
    ax.set_title("Gaussian Discriminant Analysis")
    
    # Quadratic Decision Boundary
    
    datax[:] = x[:]

    datax,norm_mean,norm_std = mean_normalization(datax)

    phi = len((np.where(datay=='Alaska'))[0])/len(datay[0])
    print(f"phi : {phi}")

    mu0 = np.sum(datax[:,(np.where(datay=='Canada'))[1]],axis = 1).reshape((datax.shape[0],1)) / len((np.where(datay=='Canada'))[1])
    print(f"mu_0 : {mu0}")

    mu1 = np.sum(datax[:,(np.where(datay=='Alaska'))[1]],axis = 1).reshape((datax.shape[0],1)) / len((np.where(datay=='Alaska'))[1])
    print(f"mu_1 : {mu1}")

    diff = datax
    diff[:,(np.where(datay=='Canada'))[1]] = diff[:,(np.where(datay=='Canada'))[1]] - mu0
    diff[:,(np.where(datay=='Alaska'))[1]] = diff[:,(np.where(datay=='Alaska'))[1]] - mu1


    sigma0 = np.dot(diff[:,(np.where(datay=='Canada'))[1]],diff[:,(np.where(datay=='Canada'))[1]].T) / len((np.where(datay=='Canada'))[1])
    sigma1 = np.dot(diff[:,(np.where(datay=='Alaska'))[1]],diff[:,(np.where(datay=='Alaska'))[1]].T) / len((np.where(datay=='Alaska'))[1])

    print(f"sigma_0 : {sigma0}")
    print(f"sigma_1 : {sigma1}")

    datax[:] = x[:]

    x1 = (np.linspace(-4,4,100)) 
    x2 = (np.linspace(-4,4,100)) 
    x1, x2 = np.meshgrid(x1, x2)
    z = np.zeros_like(x1)

    sigma0inv = pinv(sigma0)
    sigma1inv = pinv(sigma1)

    intercept = (np.log(phi/(1-phi))) - 0.5 * (np.log(det(sigma1)/det(sigma0)))

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            x = np.array([x1[i,j], x2[i,j]]).reshape((2,1))
            z[i,j] = intercept - 0.5 * np.dot((x-mu1).T,np.dot(sigma1inv,x-mu1)) + 0.5* np.dot((x-mu0).T,np.dot(sigma0inv,x-mu0))

    

    ax.scatter(datax[0,(np.where(y==1))[1]],datax[1,(np.where(y==1))[1]],c = 'red',marker = 'x',label  = 'Alaska')
    ax.scatter(datax[0,(np.where(y==0))[1]],datax[1,(np.where(y==0))[1]],c = 'blue',label = 'Canada')

    cont = ax.contour(x1 * norm_std[0] + norm_mean[0],x2 * norm_std[1] + norm_mean[1],z,0,cmap = cm.plasma)
    fig.savefig("GDA.png")
    plt.close()

    # Test Data

    X = pd.read_csv(f"{test_data_directory}/X.csv",header=None)

    x_data = X.to_numpy()
    x_data = x_data.T
    x = (x_data-norm_mean)/norm_std

    def predict(x):
        p1 = (phi/np.sqrt(det(sigma1)))*np.exp(-0.5 * np.dot((x-mu1).T,np.dot(sigma1inv,x-mu1)))
        p0 = ((1-phi)/np.sqrt(det(sigma0)))*np.exp(-0.5 * np.dot((x-mu0).T,np.dot(sigma0inv,x-mu0)))

        if(p1>p0):
            return "Alaska"
        else:
            return "Canada"
    y = []
    for i in range(x.shape[1]):
        y.append(predict(x[:,i].reshape((x.shape[0],1))))
    with open('result_4.txt', 'w+') as f:
        for items in y:
            f.write('%s\n' %items)

if __name__ == '__main__':
    main()
    