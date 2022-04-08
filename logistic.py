import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  

class Logistic:

    def __init__ (self, w, b):
        self.w=w
        self.b=b
        pass

    def sigmoid(self,z):
        '''The range of inputs is the set of all Real Numbers and the range of outputs is between 0 and 1.
        z should increase positive infinity when the output get closer to 1'''

        sig_func = 1/(1 + np.exp(-z))
        return sig_func

    def prep(self, X, w):
        '''
            Returns 1D array of probabilities
            that the class label == 1
        '''
        z = self.sigmoid(np.dot(X, w))
        return z

    def cost_function (self, w, b, X, Y):
        """computing the cost function for the Logistic regression
        M(w)"""
        m = X.shape[0]
    
    #Prediction
        final_result = self.sigmoid(np.dot(w,X.T)+b)
        Y_T = Y.T
        cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))

        return cost


    def model_train(self, X, Y, w, learning_rate, iters):
        """ Model training """
        cost_history = []

        for i in range(iters):
            w = self.gradient_cost(X, Y, w, learning_rate)

            cost = self.cost_function(X, Y, w)
            cost_history.append(cost)

        return w, cost_history