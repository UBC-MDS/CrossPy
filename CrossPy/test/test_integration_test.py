import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

import numpy as np
import pandas as pd
from CrossPy.CrossPy import cross_validation, summary_cv
from sklearn.linear_model import LinearRegression


# integration test

def test_summary_cv_AND_cross_validation_Perfect_linear():
    X, y = data_gen(nrows=100, Non_perfect=False)
    scores = cross_validation(lm(), X=X, y=y, shuffle=False)
    summary = summary_cv(scores)
    assert summary['mean'] == 1 and summary['median'] == 1 and summary['sd'] == 0, 'Perfect linear relation test does not give correct summary'

def test_summary_cv_AND_cross_validation_NonPerfect_linear():
    X, y = data_gen(nrows=100, Non_perfect=True)
    scores = cross_validation(lm(), X=X, y=y, shuffle=False)
    summary = summary_cv(scores)
    assert summary['mean'] < 1 and summary['median'] < 1 and summary['sd'] > 0, 'Non-perfect linear relation test does not give correct summary'





# Helper function

## Data Generation
def data_gen(nrows=100, Non_perfect = False):
    '''
    Generate data

    input:
    ------
    nrows: number of rows, an integer
    Non_perfect: whether X, y are perfectly linear, True or False

    output:
    ------
    X, a dataframe with nrows and two columns
    y, a dataframe with nrows and one column
    '''
    tmp_data = {"X0": range(nrows), "X1": np.random.rand(nrows)}
    X = pd.DataFrame(tmp_data)

    if Non_perfect == False:
        tmp_data = {"y": range(nrows)}
        y = pd.DataFrame(tmp_data)
    else:
        tmp_data = {"y": X.X0 + X.X1*2 + nrows/20 * np.random.randn(nrows)}
        y = pd.DataFrame(tmp_data)


    return X, y

## linear regression model
def lm():
    lm = LinearRegression()
    return lm