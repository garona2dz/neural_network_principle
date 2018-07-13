import numpy as np
from activate_util import *

def linear_backward(dZ, cache):
    '''
    implement the linear portion of backward propagation for one leayer

    Arguments:
    dZ -- Gradient of cost with respect to activtion
    cache -- tuple of value of current layer

    Returns:
    dA_prev -- Gradient of cost with respect to the activation
    dW -- Gradient of ccost with respect to W
    db -- gradient of previous layer bias
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    '''
    Implement the backward propagation for linear->activtion layer.

    Arguments:
    dA -- Gradient of current activation layer
    cache -- tuple of value current layer
    activation -- the activation function  of current layer

    Returns:
    dA_prev -- gradient of current layer
    dW -- gradient of the W
    db -- gradient of the bias
    '''
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)


    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)


    return dA_prev, dW, db

def L_model_backward(AL,Y,caches):
    '''
    implement the backward propagation for ential network

    Arguments:
    AL -- problitity vector/output
    Y -- true 'label',
    caches -- list of whole network value

    Returns:
    grads -- A dictionary with gradients
             grads['dA'+str(1)]=...
             ...
    '''
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache=caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    
    for l in reversed(range(L-1)):
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        # 这里的下标有点容易弄错，注意传入的第一个参数是下一层的dA(按照前向传播的方向看)
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, 'relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters,grads,learning_rate):
    '''
    Update parameters using gradient

    Arguments:
    parameters -- python dictionary of paramenters network
    grads -- python dictionary of gradient value

    Returns:
    paramenters -- python dictionary of your update paramenters
                   parameters['W'+str(1)]=...
                   ...
    '''
    L=len(parameters)//2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


    
