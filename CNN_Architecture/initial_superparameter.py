import numpy as np
#初始化网络超参数，网络层数、每层神经元个数


def initialize_superparameters_net(layer_dims):
    '''
    Arguments:
    layer_dims --python array (list) containing the demensions of each layer in network

    Returns:
    parameters --python dictionary containing parameters "W1,b1,W2,b2..."
    '''
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],
                                                 layer_dims[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape ==
               (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


