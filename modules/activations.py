import numpy as np
import pandas as pd
from sklearn import preprocessing

#activation functions for activating output cells of lstm. We'll require sigmoid, tanh and their derivatives dsigmoid and dtanh
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y


