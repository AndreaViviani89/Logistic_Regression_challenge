import numpy as np 
from typing import Any


class Logistic:
    """Logistic regression is a generalized linear model that 
      uses to model or predict categorical outcome variables.
      It predicts using the binary classification of 0, and 1 
      for probability of of occurring  and not occurring:
       
       (method): logistic_regression(features,target, learning_rate, number_steps, add_intercept) 

       features:   The features for the trained set from the dataset
       target:  The population of the dataset is required to obtain an important info
                for prediction
       learnin_rate:    Is a float value from the range of (0,1) with the default= 0.5, the smaller the 
                        higher the precision of accuracy of the model.
        add_intercept:  A bool function with the default= False
        number_steps:  The number iteration for the data i.e how many simulation 
                        default=100
    """

    def __init__(self,number_steps=100, learning_rate=0.5, add_intercept = False) -> Any:
        self.number_steps=number_steps
        self.learning_rate=learning_rate
        self.add_intercept=add_intercept
    

    def sigmoid(self,scores):
        '''the sigmoid function is:
            z=1 / (1 + e^x)
            The range of inputs is the set of all Real Numbers and the range of
             outputs is between 0 and 1.
            z should increase positive infinity when the output get closer to 1
        '''

        sig_func = 1/(1 + np.exp(-scores))
        return sig_func

    def __prep(self, features, weights):
        '''
            Returns 1D array of probabilities
            that the class label == 1
        '''
    
        z = self.sigmoid(np.dot(features, weights))
        return z
        
    

    def cost_function (self, features, target):
        """computing the cost function for the Logistic regression
        M(w)"""
        if self.add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
        
        weights = np.zeros(features.shape[1])
        m = features.shape[0]
    
        #Prediction
        final_result = self.sigmoid(np.dot(weights,features.T)+self.add_intercept)
        Y_T = target.T
        cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))

        return cost

    def __gradient_cost(self, features, target):
        """
            finding the gradient of the cost function
        """
    
        N = features.shape[0]
        weights = np.zeros(features.shape[1])
        #1 - Get Predictions
        predictions = self.__prep(features, weights)


        gradient = np.dot(features.T,  predictions - target) 

        #3 Take the average cost derivative for each feature
        gradient /= N

        #4 - Multiply the gradient by our learning rate
        gradient *= self.learning_rate

        #5 - Subtract from our weights to minimize cost
        weights -= gradient

        return weights

    def model_train(self, features, target):
        """ Model training with the train set i.e X_train and Y_train
        the learning rating is float number """
        cost_history = []

        for step in range(self.number_steps):
            weights= self.__gradient_cost(features, target)

            cost = self.cost_function(features, target)
            cost_history.append(cost)
        if step % 10000 == 0:
            print(self.cost_function(features, target))
        return weights, cost_history


    # def predict(self, X_pred):
    #     weights= self.__gradient_cost(self.features, self.target)
    #     final_pred=self.__prep(self, self.features, weights)
    #     y_pred = np.zeros((1,X_pred.shape[0]))
    #     for i in range(final_pred.shape[1]):
    #         if final_pred[0][i] > 0.5:
    #             y_pred[0][i] = 1
    #     return y_pred


    def accuracy(self, predicted_labels, actual_labels):
        diff = predicted_labels[0] - actual_labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))