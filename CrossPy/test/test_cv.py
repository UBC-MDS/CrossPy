import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

import pytest
import numpy as np
import pandas as pd
from CrossPy.CrossPy import cross_validation, split_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import types

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
        cross_validation(lm(), X = X, y = "y")

def test_k_as_number():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X = X, y = y, k = '3')

def test_shuffle_as_boolean_not_string():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X = X, y = y, shuffle = '1')

def test_shuffle_as_boolean_not_numeric():
    X, y = data_gen()
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

def test_k_range_k_not_one():
    X, y = data_gen(nrows = 5)
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y, k = 1)


def test_k_range_k_not_larger_than_nrows():
    X, y = data_gen(nrows = 5)
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y, k = 40)


def test_random_state_range():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X=X, y=y, random_state=-10)


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
        cross_validation(lm(), X=X, y=y, k = 2)


# Output Errors


def test_X_y_perfect_linear():
    X, y = data_gen(nrows = 100, Non_perfect=False)
    function_scores = cross_validation(lm(), X = X, y = y, shuffle = False)
    assert np.mean(function_scores) == 1 , "Testing perfect linear case should have perfect socres (score=1)"

def test_X_y_perfect_linear_random_state_12345():
    X, y = data_gen(nrows=100, Non_perfect=False)
    function_scores = cross_validation(lm(), X=X, y=y, shuffle=True, random_state=12345)
    assert np.mean(function_scores) == 1, "shuffle=True, randome state should have no effect"

def test_X_y_perfect_linear_random_state_None():
    X, y = data_gen(nrows=100, Non_perfect=False)
    function_scores = cross_validation(lm(), X=X, y=y, shuffle=True, random_state=None)
    assert np.mean(function_scores) == 1, "shuffle=True, randome state should have no effect"

def test_X_y_not_perfect_linear():
    X, y = data_gen(nrows = 100, Non_perfect=True)
    function_scores = cross_validation(lm(), X = X, y = y, shuffle = False)
    assert np.mean(function_scores) > 0 and np.mean(function_scores) < 1 , "results doesn't match sklearn"

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

def test_split_data_output_is_generator():
    X, y = data_gen()
    indices = split_data(X, shuffle=False)
    assert isinstance(indices, types.GeneratorType), "split_data output should be a generator"

def test_split_data_shuffle_False_first_val_fold():
    X, y = data_gen()
    indices = split_data(X, shuffle=False)
    ind_val, ind_train = next(indices)
    assert np.array_equal(ind_val, np.arange(34)), "`shuffle=False` doesn't work in split_data for the first validation fold"

def test_split_data_shuffle_False_last_val_fold():
    X, y = data_gen()
    indices = split_data(X, shuffle=False)
    next(indices)
    next(indices)
    ind_val, ind_train = next(indices)
    assert np.array_equal(ind_val, np.arange(67, 100)), "`shuffle=False` doesn't work in split_data for the validation fold"

def test_split_data_shuffle_True():
    X, y = data_gen()
    indices = split_data(X, shuffle=True)
    ind_val, ind_train = next(indices)
    assert not np.array_equal(ind_val, np.arange(34)), "`shuffle=True` doesn't work in split_data"



