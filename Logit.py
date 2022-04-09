
import numpy as np

class LogisticRegresion:
    """"""

    def __init__(self,number_steps=100, learning_rate=0.5, add_intercept = False) -> None:
        self.number_steps=number_steps
        self.learning_rate=learning_rate
        self.add_intercept=add_intercept

    def __sigmoid(self, scores):
        '''the sigmoid function is:
        1 / (1 + e^x)
        The range of inputs is the set of all Real Numbers 
        and the range of outputs is between 0 and 1.
        z should increase positive infinity when
         the output get closer to 1
         '''
        sig_func = 1/(1 + np.exp(-scores))
        return sig_func

    def __log_likelihood(self, features, target, weights):
        """
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
            preds=self.__sigmoid(scores)
            error=target-preds
            gradient=np.dot(features.T, error)
            weights=weights+self.learning_rate*gradient

        
            # Print log-likelihood every so often
        if step % 10000 == 0:
            print(self.__log_likelihood(features, target, weights))
        
        return weights