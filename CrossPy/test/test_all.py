import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


import pytest
import numpy as np
import pandas as pd

from CrossPy.CrossPy import train_test_split, cross_validation, summary_cv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


## Data Generation

def data_gen(nrows=100):
    '''
    Generate data

    '''
    tmp_data = {"X0": range(nrows), "X1": np.random.rand(nrows)}
    X = pd.DataFrame(tmp_data)

    tmp_data = {"y": range(nrows)}
    y = pd.DataFrame(tmp_data)

    return X, y

def lm():
    lm = LinearRegression()
    return lm
