import numpy as np
import pandas as pd
import pytest
from CrossPy.CrossPy import train_test_split

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


# Data Generation

def data_gen(nrows=100):
    """
    Generate data

    input:
    ------
    nrows: number of rows, an integer

    output:
    ------
    X, a dataframe with nrows and two columns
    y, a dataframe with nrows and one column

    """
    tmp_data = {"X0": range(nrows), "X1": np.random.rand(nrows)}
    X = pd.DataFrame(tmp_data)

    tmp_data = {"y": range(nrows)}
    y = pd.DataFrame(tmp_data)

    return X, y


#'Tests for function `train_test_split(X, y, test_size=0.25, shuffle=True, random_state=None)`
# Input Type Errors

def test_X_as_dataframe():
    X, y = data_gen()
    with pytest.raises(TypeError):
        train_test_split(X="X", y=y)


def test_y_as_dataframe():
    X, y = data_gen()
    with pytest.raises(TypeError):
        train_test_split(X=X, y="y")


def test_test_size_as_number():
    X, y = data_gen()
    with pytest.raises(TypeError):
        train_test_split(X=X, y=y, test_size='0.5')


def test_shuffle_as_boolean_not_string():
    X, y = data_gen()
    with pytest.raises(TypeError):
        train_test_split(X=X, y=y, shuffle='1')


def test_shuffle_as_boolean_not_numeric():
    X, y = data_gen()
    with pytest.raises(TypeError):
        train_test_split(X=X, y=y, shuffle=1.0)


def test_random_state_as_number():
    X, y = data_gen()
    with pytest.raises(TypeError):
        train_test_split(X=X, y=y, random_state='10')


# Input Value Errors

def test_test_size_range_large_than_1():
    X, y = data_gen()
    with pytest.raises(ValueError):
        train_test_split(X=X, y=y, test_size=1.1)


def test_test_size_range_small_than_0():
    X, y = data_gen()
    with pytest.raises(ValueError):
        train_test_split(X=X, y=y, test_size=- 0.1)


def test_random_state_range():
    X, y = data_gen()
    with pytest.raises(ValueError):
        train_test_split(X=X, y=y, random_state=-10)


# Input Dimension Errors

def test_X_y_match():
    X, y = data_gen()
    y = y[0:90]
    with pytest.raises(ValueError):
        train_test_split(X=X, y=y)


def test_y_one_column():
    X, y = data_gen()
    y = X
    with pytest.raises(ValueError):
        train_test_split(X=X, y=y)


def test_X_y_Nrows():
    X, y = data_gen(nrows=2)
    with pytest.raises(ValueError):
        train_test_split(X=X, y=y)


# Output Errors

def test_dimension_match():
    X, y = data_gen(nrows=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    assert ((X_train.shape[0] + X_test.shape[0]) == X.shape[0] and (y_train.shape[0] + y_test.shape[0]) == y.shape[
        0]), "total rows of X/y_train and Xy_test doesn't match nrows of X/y"


def test_index_match():
    X, y = data_gen(nrows=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    v1 = X_train.append(X_test).X0.sort_values().as_matrix()
    v2 = X.X0.as_matrix()
    v3 = y_train.append(y_test).y.sort_values().as_matrix()
    v4 = y.as_matrix().flatten()
    assert (np.array_equal(v1, v2) and np.array_equal(v3, v4)), "X/y_train + X/y_test is not a complete set of X/y"


def test_shuffle_True():
    X, y = data_gen(nrows=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    v1 = X_train.append(X_test).X0.as_matrix()
    v2 = X.X0.as_matrix()
    v3 = y_train.append(y_test).y.as_matrix()
    v4 = y.as_matrix().flatten()
    assert ((not np.array_equal(v1, v2)) and (
        not np.array_equal(v3, v4))), "X/y_train + X/y_test is not shuffled while `shuffle=True`"


def test_shuffle_False():
    X, y = data_gen(nrows=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    v1 = X_train.append(X_test).X0.as_matrix()
    v2 = X.X0.as_matrix()
    v3 = y_train.append(y_test).y.as_matrix()
    v4 = y.as_matrix().flatten()
    assert (np.array_equal(v1, v2) and np.array_equal(v3, v4)), "X/y_train + X/y_test is shuffled while `shuffle=False`"


def test_shuffle_False_random_state_effect():
    X, y = data_gen(nrows=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, random_state=12345)

    v1 = X_train.append(X_test).X0.as_matrix()
    v2 = X.X0.as_matrix()
    v3 = y_train.append(y_test).y.as_matrix()
    v4 = y.as_matrix().flatten()
    assert (np.array_equal(v1, v2) and np.array_equal(v3, v4)), "`shuffle=False`: random state should have no effect"
