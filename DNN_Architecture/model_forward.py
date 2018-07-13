import numpy as np
from activate_util import *
# 模型前向传播过程


def linear_forward(A, W, b):
    '''
    linear forward propagation implementation
    Arguments:
    A --previous layer activate value
    W --current layer weights of linear function 
    b --current layer weight of bias value

    Returns:
    Z -- current linear value after computing
    cache --current layer linear value
    '''

    Z = np.dot(W, A)+b

    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    '''
    LINEAR->ACTIVATION layer implementation

    Arguments:
    A_prev -- previous layer's activation value
    W -- current layer's weight 
    b -- current layers bias
    activation -- activate function

    Return:
    A -- current layer's activate value
    cache -- current linear value and activate value
    '''
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)

    return A, cache

def L_model_forward(X,parameters):
    '''
    forward propagation for L layer model network

    Arguments:
    X -- input data, numpy array of shape (example shape, number of examples)
    parameters -- model's parameters W and b, from initial_superparameters()

    Returns:
    AL -- output densor
    caches -- contatining all cache W,b,z
    '''
    caches = []
    A=X
    L=len(parameters)//2

    for l in range(1,L):
        A_prev = A
        A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation="relu")
        caches.append(cache)

    AL,cache=linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation="sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL,caches

def compute_cost(AL,Y):
    '''
    cost function implementation 

    Arguments:
    AL -- output vector corresponding to label predictions
    Y -- true "lable" vector 

    Returns:
    cost -- distance between prediction and label
    '''
    m=Y.shape[1]

    cost = -1/m*np.sum(Y*np.log(AL)+(1-Y)*(np.log(1-AL)))

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost



        
