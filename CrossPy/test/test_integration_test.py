import numpy as np
import pandas as pd
from CrossPy.CrossPy import train_test_split, cross_validation, summary_cv
from sklearn.linear_model import LinearRegression
import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


# integration test

def test_integration_perfect_linear():
    X, y = data_gen(nrows=100, Non_perfect=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scores = cross_validation(lm(), X=X_train, y=y_train, shuffle=False)
    summary = summary_cv(scores)
    assert summary['mean'] == 1 and summary['median'] == 1 and summary[
        'sd'] == 0, 'Perfect linear relation test does not give correct summary'


def test_integration_nonperfect_linear():
    X, y = data_gen(nrows=100, Non_perfect=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scores = cross_validation(lm(), X=X_train, y=y_train, shuffle=False)
    summary = summary_cv(scores)
    assert summary['mean'] < 1 and summary['median'] < 1 and summary[
        'sd'] > 0, 'Non-perfect linear relation test does not give correct summary'


# Helper function

# Data Generation
def data_gen(nrows=100, Non_perfect=False):
    """
    Generate data

    input:
    ------
    nrows: number of rows, an integer
    Non_perfect: whether X, y are perfectly linear, True or False

    output:
    ------
    X, a dataframe with nrows and two columns
    y, a dataframe with nrows and one column
    """
    tmp_data = {"X0": range(nrows), "X1": np.random.rand(nrows)}
    X = pd.DataFrame(tmp_data)

    if not Non_perfect:
        tmp_data = {"y": range(nrows)}
        y = pd.DataFrame(tmp_data)
    else:
        tmp_data = {"y": X.X0 + X.X1 * 2 + nrows / 20 * np.random.randn(nrows)}
        y = pd.DataFrame(tmp_data)

    return X, y


# linear regression model
def lm():
    """
    create an linearregression object
    :return: the object
    """
    lm = LinearRegression()
    return lm
