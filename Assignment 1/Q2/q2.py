import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import sys
plt.rcParams['agg.path.chunksize'] = 10000000000

def prediction(parameters,x):
    return np.transpose(np.dot(parameters,x.T)).reshape((x.shape[0],1))

def gradient_descent(learning_rate,parameters, y, x):
    return (learning_rate/x.shape[0])*np.sum((y-prediction(parameters,x))*x,axis=0).reshape(parameters.shape)

def loss_function(parameters, y, x):
    return (1/(2*x.shape[0]))*np.sum(np.square(y-prediction(parameters,x))).reshape((1,1))

def main():
    test_data = sys.argv[1]
    # Part a
    parameters = np.array([3,1,2]).reshape((1,3))

    x = np.ones((1000000,3))
    x[:,1] = np.random.normal(loc = 3, scale = 2, size = (1000000,))
    x[:,2] = np.random.normal(loc = -1, scale = 2, size = (1000000,))

    y = prediction(parameters,x) 

    y = y + np.random.normal(loc = 0, scale = np.sqrt(2), size = (1000000,1))

    # part b
    all_parameters = []
    learning_rate = 0.001

    all_learning_history = []

    def train(batch_size,learning_rate):
        
        num_batches = int(x.shape[0]/batch_size)
        x_batches = np.ones((num_batches,batch_size,x.shape[1]))
        y_batches = np.ones((num_batches,batch_size,y.shape[1]))

        for b in range(num_batches):
            x_batches[b] = x[((b)*batch_size):((b+1)*batch_size)]
            y_batches[b] = y[((b)*batch_size):((b+1)*batch_size)]
        parameters = np.array([0,0,0]).reshape((1,3))
        
        learning_history = []
        min_change = 1e-5
        max_iter = 10000000
        batch_round_robin = 0
        prev_epoch_loss = loss_function(parameters,y_batches[batch_round_robin],x_batches[batch_round_robin])
        current_epoch_loss = 0
        for iter in tqdm.tqdm(range(max_iter)):
            change = gradient_descent(learning_rate,parameters,y_batches[batch_round_robin],x_batches[batch_round_robin])
            
            learning_history.append(parameters)
            
            parameters = parameters+change
            
            batch_round_robin = (batch_round_robin + 1) % num_batches

            current_epoch_loss += loss_function(parameters,y_batches[batch_round_robin],x_batches[batch_round_robin])

            if batch_round_robin == 0 :
                current_epoch_loss /= num_batches
                if abs(current_epoch_loss-prev_epoch_loss)<min_change:
                    break
                prev_epoch_loss = current_epoch_loss
                current_epoch_loss = 0

        
        return parameters,learning_history

    for batch_size in [1,100,10000,1000000]:
        parameters,learning_history =  train(batch_size,learning_rate)
        all_parameters.append(parameters)
        all_learning_history.append(learning_history)
        print(f"Batch Size : {batch_size} | Learned Parameters : {parameters}")
    
    # q2test.csv
    
    # linearX = pd.read_csv(f"q2test.csv",header = None)
    # data = linearX.to_numpy()
    # x_test = np.ones((data.shape[0],3))
    # x_test[:,1] = data[:,0]
    # x_test[:,2] = data[:,1]
    # y_test = data[:,2].reshape((data.shape[0],1))
    # parameters = np.array([3,1,2]).reshape((1,3))
    # print(f"Test Loss with original hypothesis [3,1,2] = {loss_function(parameters,y_test,x_test)}")
    # iter = 0
    # min_error = [1000,0]
    # for batch_size in [1,100,10000,1000000]:
    #     error = loss_function(all_parameters[iter],y_test,x_test)
    #     print(f"Batch Size : {batch_size} | Test Loss : {error}")
    #     if error < min_error[0]:
    #         min_error = [error,iter]
    #     iter = iter+1


    # Test Dataset

    iter = 0
    min_error = [1000,0]
    for batch_size in [1,100,10000,1000000]:
        error = loss_function(all_parameters[iter],y,x)
        if error < min_error[0]:
            min_error = [error,iter]
        iter = iter+1

    linearX = pd.read_csv(f"{test_data}/X.csv",header = None)
    data = linearX.to_numpy()
    x_test = np.ones((data.shape[0],3))
    x_test[:,1] = data[:,0]
    x_test[:,2] = data[:,1]
    
    np.savetxt("result_2.txt",prediction(all_parameters[min_error[1]],x_test),fmt = '%f')

    # Learning Path
    iter = 0
    for batch_size in [1,100,10000,1000000]:

        learning_history = np.squeeze(np.array(all_learning_history[iter]))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.plot(learning_history[:,0],learning_history[:,1],learning_history[:,2])
        ax.set_xlabel("Theta_0")
        ax.set_ylabel("Theta_1")
        ax.set_zlabel("Theta_2")
        ax.set_title(f"Learning Path for Batch Size {batch_size}")
        fig.savefig(f"batch{batch_size}.png")
        plt.close()
        iter = iter+1
if __name__ == '__main__':
    main()
    