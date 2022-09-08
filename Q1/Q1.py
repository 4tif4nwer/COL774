from cProfile import label
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import pandas as pd
import tqdm
import sys

def prediction(parameters,x):
    return (np.dot(np.transpose(parameters),x)).reshape((1,x.shape[1]))


def gradient_descent(learning_rate,parameters, y, x):
    return (learning_rate/x.shape[1])*np.sum(np.multiply(y-prediction(parameters,x),x),axis=1).reshape((x.shape[0],1))

def loss_function(parameters, y, x):
    return (1/(2*x.shape[1]))*np.sum(np.square(y-prediction(parameters,x)),axis = 1)

def mean_normalization(x):
    return (x-np.mean(x,axis=1).reshape(x.shape[0],1))/np.std(x,axis=1).reshape(x.shape[0],1),np.mean(x,axis=1).reshape(x.shape[0],1),np.std(x,axis=1).reshape(x.shape[0],1)

def main():
    
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    linearX = pd.read_csv(f"{train_data}/X.csv",header=None)
    linearY = pd.read_csv(f"{train_data}/Y.csv",header=None)

    x_data = linearX.to_numpy()
    x = x_data.T
    x,norm_mean,norm_std = mean_normalization(x)
    x_full = np.ones((x.shape[0]+1,x.shape[1]))
    x_full[1,:] = x[0,:]
    x = x_full


    y = y_data = linearY.to_numpy()
    y = y.T

    learning_rate = 0.01
    min_change = 1e-15

    # print(prev_loss)
    def train(learning_rate, min_change):
        
        parameters = np.zeros((x.shape[0],1))
        learning_history = []
        current_loss = loss_function(parameters,y,x)
        prev_loss = float('inf')
        while (prev_loss-current_loss)> min_change:
            learning_history.append(([parameters[0,0],parameters[1,0],current_loss]))
            change = gradient_descent(learning_rate,parameters,y,x)
            prev_loss = current_loss
            parameters = parameters+change
            current_loss = loss_function(parameters,y,x)
        learning_history = np.squeeze(np.array(learning_history))
        return parameters,learning_history

    parameters,learning_history = train(learning_rate,min_change)

    linearX_test = pd.read_csv(f"{test_data}/X.csv",header=None)

    x_test = (linearX.to_numpy().T)
    x_test_norm = (x_test - norm_mean)/norm_std
    x_test_full = np.ones((x_test.shape[0]+1,x_test.shape[1]))
    x_test_full[1,:] = x_test[0,:]
    x_test = x_test_full
    print(prediction(parameters,x))
    np.savetxt('result_1.txt',prediction(parameters,x),fmt = "%f")
    # print("Learned Parameters : " + str(parameters))

    plt.rcParams['figure.dpi'] = 70
    plt.rcParams['figure.figsize']=[12,6]
    fig_b,ax_b = plt.subplots()
    ax_b.scatter(x_data,y_data,marker = 'o',label = "Data")

    x_axis = np.linspace(np.min(x_data),np.max(x_data),10).reshape(1,10)
    x_norm = np.squeeze(np.array([np.ones((1,10)),(x_axis-norm_mean)/norm_std]))
    y_axis = (prediction(parameters,x_norm))
    print(x_axis)

    ax_b.plot(x_axis[0],y_axis[0], c = 'red',label = "Hypothesis Function")
    ax_b.set_title("Data and Hypothesis Function")
    ax_b.set_xlabel("Acidity")
    ax_b.set_ylabel("Density")
    ax_b.legend()
    fig_b.savefig("regression_plot.png")
    plt.close()



    fig_c, ax_c = plt.subplots(subplot_kw={"projection": "3d"})
    plt.rcParams['figure.figsize']=[12,12]


    X = np.linspace(-0.5, 2, 1000)
    Y = np.linspace(-0.7, 0.7, 1000)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)

    for i in range(x.shape[1]):
            Z += np.square((X + Y*x[1,i]) - y[0,i])
    Z /= 2 * x.shape[1]

    # print(learning_history.shape)
    surf = ax_c.plot_surface(X, Y, Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)

    plot = ax_c.plot([np.squeeze(learning_history[0,0])],[np.squeeze(learning_history[0,1])],[np.squeeze(learning_history[0,2])],c = 'green', label = "Learning Path")


    def animate(nums):
        plot[0].set_data(learning_history[0:nums*15,0:2].T)
        plot[0].set_3d_properties(learning_history[0:nums*15,2].T)
        return plot


    anim = animation.FuncAnimation(fig_c, animate, 60, interval=1000/60, blit=True)
    # animate(learning_history.shape[0])
    
    fig_c.colorbar(surf, shrink=0.5, aspect=5)
    ax_c.set_xlabel("Theta_0")
    ax_c.set_ylabel("Theta_1")
    ax_c.set_title("Error Function Mesh")
    ax_c.set_zlabel("Error Function")
    ax_c.legend()
    anim.save("3Dmesh.gif", writer = PillowWriter(fps = 60))   
    plt.close()

    def contourplots(learning_rate,iter):
        
        fig, ax = plt.subplots()
        cont = ax.contour(X, Y, Z,80,cmap = cm.seismic)

        parameters,learning_history = train(learning_rate,min_change)
        print(parameters)
        plot = ax.plot([learning_history[0][0]],[learning_history[0][1]],label = "Learning Path",c = "green")


        def animate(nums):
            plot[0].set_data(learning_history[0:nums,0:2].T)
            return plot


        anim = animation.FuncAnimation(fig, animate, learning_history.shape[0]+1, interval=10, blit=True)
        animate(learning_history.shape[0])
        fig.colorbar(cont, shrink=0.5, aspect=5)
        ax.set_xlabel("Theta_0")
        ax.set_ylabel("Theta_1")
        ax.set_title(f"Error Function Contour Plot (Learning Rate = {learning_rate})")
        anim.save(f"contour{iter}.gif",dpi = 100, writer = PillowWriter(fps = 60))
        ax.legend()
        plt.close()

    contourplots(learning_rate,1)

    # Part e

    for learning_rate,iter in zip([0.001,0.025,0.1],[2,3,4]):
        contourplots(learning_rate,iter)

if __name__ == '__main__':
    main()
    