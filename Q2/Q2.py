import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

def prediction(parameters,x):
    return np.transpose(np.dot(parameters,np.transpose(x)))

def gradient_descent(learning_rate,parameters, y, x):
    return (learning_rate/x.shape[0])*np.sum((y-prediction(parameters,x))*x,axis=0)

def loss_function(parameters, y, x):
    return (1/(2*x.shape[0]))*np.sum(np.square(y-prediction(parameters,x)))


# Part a
parameters = np.array([3,1,2]).reshape((1,3))

x = np.ones((1000000,3))
x[:,1] = np.random.normal(loc = 3, scale = 2, size = (1000000,))
x[:,2] = np.random.normal(loc = -1, scale = 2, size = (1000000,))

y = prediction(parameters,x) 

y = y + np.random.normal(loc = 0, scale = np.sqrt(2), size = (1000000,1))

# part b
all_parameters = []
learning_rate = 0.01


# min_change = 0.001
all_learning_history = []

def train(batch_size,learning_rate, max_iter):
    
    num_batches = int(x.shape[0]/batch_size)
    x_batches = np.ones((num_batches,batch_size,x.shape[1]))
    y_batches = np.ones((num_batches,batch_size,y.shape[1]))

    for b in range(num_batches):
        x_batches[b] = x[((b)*batch_size):((b+1)*batch_size)]
        y_batches[b] = y[((b)*batch_size):((b+1)*batch_size)]
    parameters = np.array([0,0,0]).reshape((1,3))
    learning_history = []
    min_change = 1e-5
    print(x_batches[0])
    batch_round_robin = 0
    prev_epoch_loss = loss_function(parameters,y_batches[batch_round_robin],x[batch_round_robin])
    current_epoch_loss = 0
    for iter in tqdm.tqdm(range(max_iter)):
        change = gradient_descent(learning_rate,parameters,y_batches[batch_round_robin],x_batches[batch_round_robin])
        
        learning_history.append(parameters)
        
        parameters = parameters+change
        
        batch_round_robin = (batch_round_robin + 1) % num_batches

        current_epoch_loss += loss_function(parameters,y_batches[batch_round_robin],x[batch_round_robin])

        if batch_round_robin == 0 :
            current_epoch_loss /= num_batches
            if(abs(current_epoch_loss-prev_epoch_loss)<min_change):
                break
            prev_epoch_loss = current_epoch_loss
            current_epoch_loss = 0

    # print(parameters)
    return parameters,learning_history

for batch_size in [1,100,10000,1000000]:
    parameters,learning_history =  train(batch_size,learning_rate,10000000)
    all_parameters.append(parameters)
    all_learning_history.append(learning_history)
    print(f"Batch Size : {batch_size} | Learned Parameters : {parameters}")

# Test Dataset

linearX = pd.read_csv("q2test.csv")
data = linearX.to_numpy()
x = np.ones((data.shape[0],3))
x[:,1] = data[:,0]
x[:,2] = data[:,1]
y = data[:,2].reshape((data.shape[0],1))

print(f"Test Data Error using original hypothesis : {loss_function(np.array([3,1,2]).reshape((1,3)),y,x)}")

iter = 0
for batch_size in [1,100,10000,1000000]:
    print(f"Batch Size : {batch_size} | Test Error : " + str(loss_function(all_parameters[iter],y,x)))
    iter = iter+1


# Learning Path
iter = 0
for batch_size in [1,100,10000,1000000]:

    learning_history = np.squeeze(np.array(all_learning_history[iter]))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot(learning_history[:,0],learning_history[:,1],learning_history[:,2])
    plt.show()
    iter = iter+1
