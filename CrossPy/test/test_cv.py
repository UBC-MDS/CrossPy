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


## Tests for_cross_validation()
##'Tests for function `cross_validation(model, X, y, k = 3, shuffle = True, random_state = None)

# Input Type Errors

def test_X_as_dataframe():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X = "X", y = y)

def test_y_as_dataframe():
    with pytest.raises(TypeError('`y` must be a dataframe')):
        cross_validation(lm, X = X, y = y_list)

def test_k_as_number():
    with pytest.raises(TypeError('`k` must be an integer')):
        cross_validation(lm, X = X, y = y, k = '3')

def test_shuffle_as_boolean():
    with pytest.raises(TypeError('`shuffle` must be True or False')):
        cross_validation(lm, X = X, y = y, shuffle = '1')
    with pytest.raises(TypeError('`shuffle` must be True or False')):
        cross_validation(lm, X = X, y = y, shuffle = 1)
    with pytest.raises(TypeError('`shuffle` must be True or False')):
        cross_validation(lm, X=X, y=y, shuffle=1.0)

def test_random_state_as_number():
    with pytest.raises(TypeError('`random_state` must be a number or None')):
        cross_validation(lm, X = X, y = y, random_state = '10')


# Input Value Errors

def test_k_range():
    with pytest.raises(TypeError('`k` must be an integer 2 or greater')):
        cross_validation(lm, X=X, y=y, k = 1)
    with pytest.raises(TypeError('`k` must be greater than # obs in X and y')):
        cross_validation(lm, X=X, y=y, k = 40)


def test_random_state_range():
    with pytest.raises(TypeError('`random_state` must be nonnegative')):
        cross_validation(lm, X=X, y=y, random_state=-10)


# Input Dimension Errors

def test_X_y_match():
    with pytest.raises(TypeError("dim of `X` doesn't equal dim of `y`")):
        cross_validation(lm, X=X_longer, y=y)


def test_y_one_column():
    with pytest.raises(TypeError('`y` is more than one feature')):
        cross_validation(lm, X=X, y=y_2)


def test_X_y_Nrows():
    with pytest.raises(TypeError('sample size is less than 3, too small for CV')):
        cross_validation(lm, X=X.iloc[0:2, :], y=y.iloc[0:2, :])


# Output Errors

def compare_sklearn_mod_0():
    assert max(cv_scores_mod_0 - sk_cv_score_mod_0) < 0.000001, "results doesn't match sklearn"

def compare_sklearn_mod_0():
    assert max(
        cv_scores_mod_not_0 - sk_cv_score_mod_not_0) < 0.000001, "results doesn't match sklearn"

