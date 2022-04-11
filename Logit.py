
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

class LogitRegresion:
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

    def sigmoid(self, scores):
        '''the sigmoid function is:
            z=1 / (1 + e^x)
            The range of inputs is the set of all Real Numbers 
            and the range of outputs is between 0 and 1.
            z should increase positive infinity when
            the output get closer to 1
         '''
        sig_func = 1/(1 + np.exp(-scores))
        return sig_func

    def __log_likelihood(self, features, target, weights):
        """
        The log-likelihood can be viewed as as sum over all the training data.
        the log likelihood is the logrithm of the of maximium 
        of the independent variable of the features of the data set
        of sigmoid function.
        scores = features * weights
        ll = SUM{ target * scores - log( 1 + e^x)}
        """
        scores=np.dot(features, weights)
        ll=np.sum(np.dot(target, scores)-np.log(1+np.exp(scores)))
        return ll
    
    def logistic_regression(self, features, target):
        """
        logistic_regression is to optimized the linear regression with an intercept using the 
        sigmoid function with the computation of the gradient of the 
        log likelihood of the logit function. 

        scores -> feature * weights
        preds -> sigmoid(scores)
        error = target-preds
        gradient -> transposed feature * error
        weight -> weights+ learning_rate * gradient
        
        """

        if self.add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
        
        weights = np.zeros(features.shape[1])
    
        for step in range(self.number_steps):
            # scores -> feature * weights
            # preds -> sigmoid(scores)
            # error = target-preds
            # gradient -> transposed feature * error
            # weight -> weights+ learning_rate * gradient
            scores=np.dot(features, weights)
            preds=self.sigmoid(scores)
            error=target-preds
            gradient=np.dot(features.T, error)
            weights=weights+self.learning_rate*gradient

        
            # Print log-likelihood every so often
            if step % 10000 == 0:
                print(self.__log_likelihood(features, target, weights))
        
        return weights


    def accuracy(self, features, target):

        '''We need to use the final weights to get the logits for the dataset (`final_scores`). Then I can use `sigmoid` to get the final predictions.
        -   accuracy is the percentages of correct guesses 0-1'''

        weights=self.logistic_regression(features, target)
        finalscores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                 features)), weights)
        preds = np.round(self.sigmoid(finalscores))
        accuracy = (preds == target).sum().astype(float) / len(preds)
        return accuracy


    # def graph(self, features, target):
    #     self.features = features
    #     self.target = target
    #     plt.figure(figsize=(12,8))
    #     fig1 = plt.scatter(self.features[:, 0], self.features[:, 1], c = self.target, alpha = .4)
    #     return fig1



