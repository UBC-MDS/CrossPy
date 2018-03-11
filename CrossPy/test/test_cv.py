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

    input:
    ------
    nrows: number of rows, an integer

    output:
    ------
    X, a dataframe with nrows and two columns
    y, a dataframe with nrows and one column
    '''
    tmp_data = {"X0": range(nrows), "X1": np.random.rand(nrows)}
    X = pd.DataFrame(tmp_data)

    tmp_data = {"y": range(nrows)}
    y = pd.DataFrame(tmp_data)

    return X, y

def lm():
    lm = LinearRegression()
    return lm


## Tests for_cross_validation()
##'Tests for function `cross_validation(model, X, y, k = 3, shuffle = True, random_state = None)

# Input Type Errors

def test_X_as_dataframe():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X = "X", y = y)

def test_y_as_dataframe():
    X, y = data_gen()
    with pytest.raises(TypeError):
        train_test_split(lm(), X = X, y = "y")

def test_k_as_number():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X = X, y = y, k = '3')

def test_shuffle_as_boolean():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X = X, y = y, shuffle = '1')
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y, shuffle=1.0)

def test_random_state_as_number():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X = X, y = y, random_state = '10')

def test_model_linear_regression():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation("LINEAR MODEL", X = X, y = y, random_state = 10)


# Input Value Errors

def test_k_range():
    X, y = data_gen(nrows = 5)
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y, k = 1)
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y, k = 40)


def test_random_state_range():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm, X=X, y=y, random_state=-10)


# Input Dimension Errors

def test_X_y_match():
    X, y = data_gen()
    y = y[0:90]
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y)


def test_y_one_column():
    X, y = data_gen()
    y = X
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y)


def test_X_y_Nrows():
    X, y = data_gen(nrows = 2)
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y)


# Output Errors

def test_compare_sklearn_mod_0():
    X, y = data_gen(nrows = 99)
    function_scores = cross_validation(lm(), X = X, y = y, shuffle = False)
    sklearn_scores = cross_val_score(lm(), X = X, y = y, cv = 3)
    assert max(function_scores - sklearn_scores) < 0.000001, "results doesn't match sklearn"

def test_compare_sklearn_mod_not_0():
    X, y = data_gen(nrows = 100)
    function_scores = cross_validation(lm(), X = X, y = y, shuffle = False)
    sklearn_scores = cross_val_score(lm(), X = X, y = y, cv = 3)
    assert max(function_scores - sklearn_scores) < 0.000001, "results doesn't match sklearn"

