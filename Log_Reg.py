import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  

class LogReg:

    x = None
    y = None



    def __sigmoid(self, z):
        # The range of inputs is the set of all Real Numbers and the range of outputs is between 0 and 1.
        # z should increase positive infinity when the output get closer to 1
        sig_func = (1 + np.exp(-z))/1
        return sig_func

    
