

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

def cross_validation(model, X, y, k = 3, shuffle = TRUE, random_state = None):
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
    pass

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
