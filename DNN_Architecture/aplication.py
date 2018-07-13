import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from initial_superparameter import initialize_superparameters_net
from model_forward import L_model_forward,compute_cost
from model_backward import L_model_backward,update_parameters

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#define layers_dims network
def layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    '''
    Implement a nerual network :layers_dims(n_x,n_h,n_y)

    Arguments:
    X -- imput data,of shape(n_x,number of examples)
    Y -- true 'label' vector
    layers_dims -- dimensions of the layers (n_x,n_h,n_y)
    num_iterations -- number of iterations of optimazation loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- print situation of optimization

    Returns:
    parmeters -- final parameters after optimization
    '''
    costs = []    

    parameters = initialize_superparameters_net(layers_dims)

    for i in range(0,num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters,  grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) 
    p = np.zeros((1,m))
    

    probas, caches = L_model_forward(X, parameters)

    

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    

    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p
        
def test_image(image_path):
    '''
    coding this function for predict your image ,good luck
    '''
    pass

def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
    
    parameters=layer_model(train_x,train_y,layers_dims,learning_rate=0.005,num_iterations=2500,print_cost=True)

    pred_train=predict(train_x,train_y,parameters)

    pred_test=predict(test_x,test_y,parameters)
    
    #test
    test_image('')

if __name__ == '__main__':
    main()
    