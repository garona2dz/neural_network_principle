import numpy as np
#模型前向传播过程
def linear_forward(A,W,b):
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

    Z=np.dot(W,A)+b

    assert(Z.shape==(W.shape[0],A.shape[1]))

    cache=(A,W,b)

    return Z,cache

