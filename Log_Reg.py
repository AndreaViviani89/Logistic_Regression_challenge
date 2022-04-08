from tkinter import X, Y
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  

class LogReg:

    x = None
    y = None



    def __sigmoid(self, z):
        '''The range of inputs is the set of all Real Numbers and the range of outputs is between 0 and 1.
        z should increase positive infinity when the output get closer to 1'''
        #   Sigmoid formula 1/(1 + e−t).

        sig_func = 1/(1 + np.exp(-z))
        self.sig_func = sig_func
        return sig_func



    def __cost_function(self, ):
        y_pred = self.sig_func
        cost = -(1/X.shape[0])*np.sum(Y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
        