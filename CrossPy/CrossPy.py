import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_random_state



def train_test_split(X, y, test_size = 0.25, shuffle = True, random_state = None):
    '''
    split features X and target y into train and test sets

    inputs
    ------
    X: a pandas dataframe with at least 3 rows
    y: a pandas dataframe with at least 3 rows and only one column
    test_size: a float number between 0 and 1, the fraction for test size
    shuffle: boolean, if True --> shuffle the rows before splitting
    random_state: a positive integer number or None, for setting the random state

    returns:
    --------
    X_train: a pandas dataframe, subset of X
    X_test: a pandas dataframe, subset of X except X_train
    y_train: a pandas dataframe, subset of y
    y_test: a pandas dataframe, subset of y except y_train

    '''
    pass




def cross_validation(model, X, y, k = 3, shuffle = True, random_state = None):
    '''
    Perform cross validation on features X and target y using the model

    inputs
    ------
    model: a model object from sklearn
    X: a pandas dataframe with at least 3 rows
    y: a pandas dataframe with at least 3 rows and only one column
    k: the No. of folds for cross validation
    shuffle: boolean, if True --> shuffle the rows before splitting
    random_state: a positive integer number or None, for setting the random state

    returns:
    --------
    scores: a vector of validation scores
    '''
    ## Input Errors
    if not isinstance(X, pd.DataFrame):
        raise TypeError('`X` must be a dataframe')
    if not isinstance(y, pd.DataFrame):
        raise TypeError('`y` must be a dataframe')
    if not isinstance(k, int):
        raise TypeError('`k` must be an integer')
    if not isinstance(shuffle, bool):
        raise TypeError('`shuffle` must be True or False')
    if not (isinstance(random_state, int) or isinstance(random_state, float) or random_state is None):
        raise TypeError('`random_state` must be a number or None')
    if not (random_state is None):
        if random_state <= 0:
            raise TypeError('`random_state` must be nonnegative')
    if k < 2 or k > X.shape[0]:
        raise TypeError('`k` must be an integer 2 or greater, and be less than # obs in X and y')
    if X.shape[0] != y.shape[0]:
        raise TypeError("dim of `X` doesn't equal dim of `y`")
    if y.shape[1] != 1:
        raise TypeError('`y` is more than one feature')
    if X.shape[0] < 3:
        raise TypeError('sample size is less than 3, too small for CV')
    if type(model) != type(LinearRegression()):
        raise TypeError('model is not an sklearn LinearRegression model')


    # If shuffle is True or False, this code will still run - see helper function below
    indices = split_data(X, k, shuffle, random_state)

    # Initialize scores output
    scores = np.arange(k)
    # For each fold tuple, get the corresponding training and val X and y, then train and score each
    for i in np.arange(k):
        ind_train, ind_val = next(indices)
        X_train = X.iloc[ind_train, :]
        X_val = X.iloc[ind_val, :]
        y_train = X.iloc[ind_train, :]
        y_val = X.iloc[ind_val, :]

        # Fit each model and score with R^2
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores[i] = score

    return scores


def split_data(X, k=3, shuffle=True, random_state = None):
    """
    Helper Function for cross_validaton
    :param X: dataframe
    :param k: number of splits
    :param shuffle: True or False for whether to shuffle indicies
    :param random_state: default None
    :return: yields a 2-tuple of np.array indices, the first for validation indicies, the second for training.
            yields 2-tuple for each k folds (eg: k = 3, would have 3 2-tuples)

    with help from code: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/model_selection/_split.py#L357

    """

    all_indices = np.arange(X.shape[0])
    n_samples = X.shape[0]

    ## Should initialize what the output will be here.. based on the k there should be indices for X and y rows,
    ## therefore could have a list of tuples??

    # In order to align with scikit learn - cross referenced: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/model_selection/_split.py#L357
    # The first n_samples % n_splits folds have size n_samples // n_splits + 1,
    # other folds have size n_samples // n_splits, where n_samples is the number of samples.

    fold_extra = n_samples % k
    num_folds = 0

    fold_sizes = (n_samples // k) * np.ones(k, dtype=np.int)
    fold_sizes[0:fold_extra] += 1

    val_indicies = []

    if shuffle == True:
        check_random_state(random_state).shuffle(all_indices)

    current = 0
    for fold in fold_sizes:
        start, stop = current, current + fold
        val_indicies.append(all_indices[start:stop])
        current = stop

    for fold in val_indicies:
        val = fold
        other = np.setdiff1d(all_indices, val)
        yield (val, other)


def summary_cv(scores):
    '''
    Calculate statistics of cross validation cv_scores

    inputs
    ------
    scores: a vector of validation scores

    returns:
    -------
    mean: mean of CV scores
    standard_deviation: standard_deviation of CV scores
    mode: mode of CV scores
    median: median of CV scores
    '''
    pass

