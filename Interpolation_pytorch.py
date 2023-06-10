# Interpolation using Neural Networks from scratch 
import torch 
import matplotlib.pyplot as plt  
import numpy as np 
from math import pi 

# PROBLEM : The NN output is between [0,1] so I can only approximate functions on 
# this range. This is important because it will also happen in PINNs. 
# Read about PINNs implementation. I should probably use a SGD algorithm that is 
# already on pytorch. 

# Function that we want to interpolate 
def Interpolation_function(x):
    return 0.5*torch.sin(x)+0.5
    # return torch.tensor(x)

# Interval in which we are interpolating and number of nodes
I=[0,2*pi] ; N=10; 
# I=[0,1] ; N=10; 

# We generate our training data
X_train = torch.linspace(I[0],I[1],N)
X_train.reshape(len(X_train),1)
Y_train = Interpolation_function(X_train) 

# The loss function is the MSE
def loss_function(Y_estimation,Y_label):
    return sum((Y_estimation-Y_label)**2)/len(Y_label)

# NN with one hidden layer 
NL=[10,5,10] # Number of neurons per layer
# Defining the matrices for each layer, initialized with normal random values 
w1=torch.tensor(np.random.normal(size=(NL[1],NL[0])),requires_grad=True,dtype=torch.float)
w2=torch.tensor(np.random.normal(size=(NL[2],NL[1])),requires_grad=True,dtype=torch.float)
W=[w1,w2] # We have all the weights here 
# Defining the biasses for each layer, initialized with normal random values
b1=torch.zeros(NL[1],requires_grad=True,dtype=torch.float)
b2=torch.zeros(NL[2],requires_grad=True,dtype=torch.float)
b=[b1,b2] # We have all the biasses here 
# Defining the activation slopes 

# We calculate the NN output 
def Neural_Network(W,b,NL,X):
    Nz=X
    for k in [0,1]:
        Nz=torch.sigmoid(torch.mv(W[k],Nz)+b[k])
    return Nz 

# Gradient descent in order to obtain w and b that minimizes the loss_function
lr=1 # learning rate for gradient descent algorithm 
max_iter=5000 # maximum number of iterations 
L=[] # list where er will append the lost function values
for k in range(0,max_iter):
    Y_estimation=Neural_Network(W,b,NL,X_train)
    loss=loss_function(Y_estimation,Y_train)
    loss.backward()
    L.append(loss)
    with torch.no_grad(): # Does not affect the gradient
        # Update the weights and biasses
        W[0]-=lr*W[0].grad ; b[0]-=lr*b[0].grad
        W[1]-=lr*W[1].grad ; b[1]-=lr*b[1].grad
        # Reset gradients
        W[0].grad.zero_() ; b[0].grad.zero_()
        W[1].grad.zero_() ; b[1].grad.zero_() 
    print('Iteration : ',k,'. Loss function : ', L[-1])

# # Test data 
# X_test=X_train+(I[1]-I[0])/(2*N)
# Y_test=Interpolation_function(X_test)

# # Calculate test results 
# Y_test_estimation=Neural_Network(W,b,NL,X_test) 

# I do not know if that makes sense, because in PINNs I just need to evaluate u at 
# the inner training points. I guess...

# Plot training and testing results 
with torch.no_grad():
    plt.plot(X_train,Y_train,'o')
    plt.plot(X_train,Y_estimation,'x')
    # plt.plot(X_test,Y_test,'o')
    # plt.plot(X_test,Y_test_estimation,'x')
    plt.legend(['Train data','NN train results'])
    # plt.legend(['Train data','NN train results','Test data','NN test results'])