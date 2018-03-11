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

## Tests for train_test_split()
## `train_test_split(X, y, test_size = 0.25, shuffle = True, random_state = None)


# Input Type Errors
def test_X_as_dataframe():
    X, y = data_gen()
    with pytest.raises(TypeError):
        cross_validation(lm(), X = "X", y = y)

def test_y_as_dataframe():
    X, y = data_gen()
    with pytest.raises(TypeError):
        train_test_split(lm(), X = X, y = "y")

def test_test_size_as_number():

    with pytest.raises(TypeError('`test_size` must be a number')):
        train_test_split(X = X, y = y, test_size = '0.25')

def test_shuffle_as_boolean():
    with pytest.raises(TypeError('`shuffle` must be True or False')):
        train_test_split(X = X, y = y, shuffle = '1')
    with pytest.raises(TypeError('`shuffle` must be True or False')):
        train_test_split(X = X, y = y, shuffle = 1)
    with pytest.raises(TypeError('`shuffle` must be True or False')):
        train_test_split(X=X, y=y, shuffle=1.0)

def test_random_state_as_number():
    with pytest.raises(TypeError('`random_state` must be a number or None')):
        train_test_split(X = X, y = y, random_state = '10')


# Input Value Errors

def test_test_size_range():
    with pytest.raises(TypeError('`test_size` must be between 0 and 1')):
        train_test_split(X = X, y = y, test_size = 2)
    with pytest.raises(TypeError('`test_size` must be between 0 and 1')):
        train_test_split(X = X, y = y, test_size = -1)

def test_random_state_range():
    with pytest.raises(TypeError('`random_state` must be nonnegative')):
        train_test_split(X = X, y = y, random_state = -10)


# Input Dimension Errors

def test_X_y_match():
    with pytest.raises(TypeError("dim of `X` doesn't equal dim of `y`")):
        train_test_split(X = X_longer, y = y)


def test_y_one_column():
    with pytest.raises(TypeError('`y` is more than one feature')):
        train_test_split(X = X, y = y_2)

def test_X_y_Nrows():
    with pytest.raises(TypeError('sample size is less than 3, too small for splitting')):
        train_test_split(X = X.iloc[0:2,:], y = y.iloc[0:2,:])


# Output Errors

def test_output_as_dataframe():
    assert isinstance(X_train, pd.DataFrame), "Output is not a dataframe"
    assert isinstance(X_test, pd.DataFrame), "Output is not a dataframe"
    assert isinstance(y_train, pd.DataFrame), "Output is not a dataframe"
    assert isinstance(y_test, pd.DataFrame), "Output is not a dataframe"

def test_output_dim_match():
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0], \
        "Sum of rows of train and test set doesn't match X"
    assert X_train.shape[1] == X.shape[1], \
        "No. of columns of train set doesn't match X"
    assert X_test.shape[1] == X.shape[1], \
        "No. of columns of test set doesn't match X"
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0], \
        "Sum of rows of train and test set doesn't match y"

def test_output_shuffle_match():
    assert X_train.equals(X.iloc[0:X_train.shape[0],:]), \
        "`Shuffle = True` doesn't work!"


def test_output_shuffle_match():
    assert X_train.equals(X.iloc[0:X_train.shape[0], :]), \
        "`Shuffle = False` doesn't work!"


## Tests for_cross_validation()
##'Tests for function `cross_validation(model, X, y, k = 3, shuffle = True, random_state = None)

# Input Type Errors

def test_X_as_dataframe():
    with pytest.raises(TypeError('`X` must be a dataframe')):
        cross_validation(lm, X = X_matrix, y = y)

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
    assert max(cv_scores_mod_not_0 - sk_cv_score_mod_not_0) < 0.000001, "results doesn't match sklearn"



## Tests summary_cv()
##'Tests for function `summary_cv(scores)

# Example Data
def gen_summary():
    list =  [0.97, 0.96, 0.98, 0.97, 0.95, 0.97]
    return list

# Input Errors:

def test_input_as_dataframe():
    with pytest.raises(TypeError):
        summary_cv(scores = pd.DataFrame(data={'cv_scores':[0.97, 0.96, 0.98, 0.97, 0.95, 0.97]}))

def test_input_as_tuple():
    with pytest.raises(TypeError):
        summary_cv(scores = (0.96, 0.97, 0.98, 0.99))

def test_input_contains_string():
    with pytest.raises(TypeError):
        summary_cv(scores = [0.96, 0.97, "0.98"])

def test_zero_length_input():
    with pytest.raises(TypeError):
        summary_cv(scores = [])

def test_input_contains_negative():
    with pytest.raises(ValueError):
        summary_cv(scores = [0.96, 0.97, -0.98])

def test_input_contains_over_1():
    with pytest.raises(ValueError):
        summary_cv(scores = [0.96, 0.97, 1.98])

# Output Errors

def test_is_dict():
    assert isinstance(summary_cv(gen_summary()), dict)

def test_output_length():
    assert len(summary_cv(gen_summary())) == 4

def test_is_float():
    assert isinstance(summary_cv(gen_summary())['mean'], float)
    assert isinstance(summary_cv(gen_summary())['median'], float)
    assert isinstance(summary_cv(gen_summary())['mode'], float)
    assert isinstance(summary_cv(gen_summary())['sd'], float)


def test_summary_cv():
    assert summary_cv(gen_summary())['mean'] == 0.967
    assert summary_cv(gen_summary())['median'] == 0.97
    assert summary_cv(gen_summary())['mode'] == 0.97
    assert summary_cv(gen_summary())['sd'] == 0.009
