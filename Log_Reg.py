# from tkinter import X, Y
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  

class LogReg:

    # x = None
    # y = None

    def __init__(self, w, b):
        self.w=w
        self.b=b

    def __sigmoid(self, z):
        '''The range of inputs is the set of all Real Numbers and the range of outputs is between 0 and 1.
        z should increase positive infinity when the output get closer to 1'''
        #   Sigmoid formula 1/(1 + e−t).

        sig_func = 1/(1 + np.exp(-z))
        self.sig_func = sig_func
        return sig_func



    def __cost_function(self, y, w, b, X, Y):
        # y_pred = self.sig_func
        # cost = -(1/X.shape[0])*np.sum(Y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
        # return cost

        #   m --> number of training examples
        m = X.shape[0]

        final_result = self.sigmoid(np.dot(w,X.T)+b)
        Y_T = Y.T
        cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
        
        return cost


    def __gradient(self,X,y):
        Dotx = np.dot(X.T, X)
        Dotx_inverse = np.linalg.inv(Dotx)
        Inverse_DotXT = np.dot(Dotx_inverse,X.T)
        final = np.dot(Inverse_DotXT, y)
        return final


    def model_train(self, X, Y, w, learning_rate, iters):
        
        """ Model training """
        cost_history = []

        for i in range(iters):
            w = self.gradient_cost(X, Y, w, learning_rate)

            cost = self.cost_function(X, Y, w)
            cost_history.append(cost)



    def predict(self,X):
        z = Dotx(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis



    def f1_score(y,y_hat):
        tp,tn,fp,fn = 0,0,0,0
        for i in range(len(y)):
            if y[i] == 1 and y_hat[i] == 1:
                tp += 1
            elif y[i] == 1 and y_hat[i] == 0:
                fn += 1
            elif y[i] == 0 and y_hat[i] == 1:
                fp += 1
            elif y[i] == 0 and y_hat[i] == 0:
                tn += 1
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*precision*recall/(precision+recall)
        return f1_score