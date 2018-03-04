import pytest
import numpy as np
import pandas as pd
#from CrossPy.CrossPy import crosspy
import CrossPy
from sklearn.linear_model import LinearRegression


# read test data
data = pd.read_csv("./test_data/test_data_short.csv")
# build test data
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_matrix = X.as_matrix()
y_list = list(y)
X_longer = X.append(X)
y_2 = pd.concat([y, y], axis=1)

# build a model
lm = LinearRegression()


class Test_train_test_split():
'''
Tests for function `train_test_split(X, y, test_size = 0.25, shuffle = True, random_state = None)`
'''

    # Input Type Errors

    def test_X_as_dataframe(self):
        with pytest.raises(TypeError('`X` must be a dataframe')):
            train_test_split(X = X_matrix, y = y)

    def test_y_as_dataframe(self):
        with pytest.raises(TypeError('`y` must be a dataframe')):
            train_test_split(X = X, y = y_list)

    def test_test_size_as_number(self):
        with pytest.raises(TypeError('`test_size` must be a number')):
            train_test_split(X = X, y = y, test_size = '0.25')

    def test_shuffle_as_boolean(self):
        with pytest.raises(TypeError('`shuffle` must be True or False')):
            train_test_split(X = X, y = y, shuffle = '1')
        with pytest.raises(TypeError('`shuffle` must be True or False')):
            train_test_split(X = X, y = y, shuffle = 1)
        with pytest.raises(TypeError('`shuffle` must be True or False')):
            train_test_split(X=X, y=y, shuffle=1.0)

    def test_random_state_as_number(self):
        with pytest.raises(TypeError('`random_state` must be a number or None')):
            train_test_split(X = X, y = y, random_state = '10')


    # Input Value Errors

    def test_test_size_range(self):
        with pytest.raises(TypeError('`test_size` must be between 0 and 1')):
            train_test_split(X = X, y = y, test_size = 2)
        with pytest.raises(TypeError('`test_size` must be between 0 and 1')):
            train_test_split(X = X, y = y, test_size = -1)

    def test_random_state_range(self):
        with pytest.raises(TypeError('`random_state` must be nonnegative')):
            train_test_split(X = X, y = y, random_state = -10)


    # Input Dimension Errors

    def test_X_y_match(self):
        with pytest.raises(TypeError("dim of `X` doesn't equal dim of `y`")):
            train_test_split(X = X_longer, y = y)


    def test_y_one_column(self):
        with pytest.raises(TypeError('`y` is more than one feature')):
            train_test_split(X = X, y = y_2)

    def test_X_y_Nrows(self):
        with pytest.raises(TypeError('sample size is less than 3, too small for splitting')):
            train_test_split(X = X.iloc[0:2,:], y = y.iloc[0:2,:])


    # Output Errors
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    def test_output_as_dataframe(self):
        assert isinstance(X_train, pd.DataFrame), "Output is not a dataframe"
        assert isinstance(X_test, pd.DataFrame), "Output is not a dataframe"
        assert isinstance(y_train, pd.DataFrame), "Output is not a dataframe"
        assert isinstance(y_test, pd.DataFrame), "Output is not a dataframe"

    def test_output_dim_match(self):
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0], \
            "Sum of rows of train and test set doesn't match X"
        assert X_train.shape[1] == X.shape[1], \
            "No. of columns of train set doesn't match X"
        assert X_test.shape[1] == X.shape[1], \
            "No. of columns of test set doesn't match X"
        assert y_train.shape[0] + y_test.shape[0] == y.shape[0], \
            "Sum of rows of train and test set doesn't match y"

    def test_output_shuffle_match(self):
        assert X_train.equals(X.iloc[0:X_train.shape[0],:]), \
            "`Shuffle = True` doesn't work!"

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    def test_output_shuffle_match(self):
        assert X_train.equals(X.iloc[0:X_train.shape[0], :]), \
            "`Shuffle = False` doesn't work!"


class Test_cross_validation():
'''
Tests for function `cross_validation(model, X, y, k = 3, shuffle = TRUE, random_state = None)`
'''

    # Input Type Errors

    def test_X_as_dataframe(self):
        with pytest.raises(TypeError('`X` must be a dataframe')):
            cross_validation(lm, X = X_matrix, y = y)

    def test_y_as_dataframe(self):
        with pytest.raises(TypeError('`y` must be a dataframe')):
            cross_validation(lm, X = X, y = y_list)

    def test_k_as_number(self):
        with pytest.raises(TypeError('`k` must be an integer')):
            cross_validation(lm, X = X, y = y, k = '3')

    def test_shuffle_as_boolean(self):
        with pytest.raises(TypeError('`shuffle` must be True or False')):
            cross_validation(lm, X = X, y = y, shuffle = '1')
        with pytest.raises(TypeError('`shuffle` must be True or False')):
            cross_validation(lm, X = X, y = y, shuffle = 1)
        with pytest.raises(TypeError('`shuffle` must be True or False')):
            cross_validation(lm, X=X, y=y, shuffle=1.0)

    def test_random_state_as_number(self):
        with pytest.raises(TypeError('`random_state` must be a number or None')):
            cross_validation(lm, X = X, y = y, random_state = '10')


    # Input Value Errors

    def test_k_range(self):
        with pytest.raises(TypeError('`k` must be an integer 2 or greater')):
            cross_validation(lm, X=X, y=y, k = 1)
        with pytest.raises(TypeError('`k` must be greater than # obs in X and y')):
            cross_validation(lm, X=X, y=y, k = 40)


    def test_random_state_range(self):
        with pytest.raises(TypeError('`random_state` must be nonnegative')):
            cross_validation(lm, X=X, y=y, random_state=-10)


    # Input Dimension Errors

    def test_X_y_match(self):
        with pytest.raises(TypeError("dim of `X` doesn't equal dim of `y`")):
            cross_validation(lm, X=X_longer, y=y)


    def test_y_one_column(self):
        with pytest.raises(TypeError('`y` is more than one feature')):
            cross_validation(lm, X=X, y=y_2)


    def test_X_y_Nrows(self):
        with pytest.raises(TypeError('sample size is less than 3, too small for CV')):
            cross_validation(lm, X=X.iloc[0:2, :], y=y.iloc[0:2, :])


    # Output Errors
    cv_scores = cross_validation(lm, X=X, y=y)

    cv_scores_sklearn = np.array([])

    def compare_sklearn():
        assert max(cv_scores - cv_scores_sklearn) < 0.000001, "results doesn't match sklearn"


class summary_cv():

    cv_scores = [0.97, 0.96, 0.98, 0.97, 0.95, 0.97]
    summary = summary_cv(cv_scores)

    # Input Errors:

    def test_input_as_dataframe(self):
        with pytest.raises(TypeError('`scores` must be a list.')):
            summary_cv(scores = pd.DataFrame(data={'cv_scores':[0.97, 0.96, 0.98, 0.97, 0.95, 0.97]}))

    def test_input_as_tuple(self):
        with pytest.raises(TypeError('`scores` must be a list.')):
            summary_cv(scores = (0.96, 0.97, 0.98, 0.99))

    def test_input_contains_string(self):
        with pytest.raises(TypeError('Elements of `scores` must be numbers.')):
            summary_cv(scores = c(0.96, 0.97, "0.98"))

    def test_zero_length_input(self):
        with pytest.raises(DimensionError('`scores` cannot be of length zero.')):
            summary_cv(scores = [])

    def test_input_contains_negative(self):
        with pytest.raises(ValueError('`scores` must be a nonnegative number.')):
            summary_cv(scores = [0.96, 0.97, -0.98])

    def test_input_contains_over_1(self):
        with pytest.raises(ValueError('`scores` must be between 0 and 1.')):
            summary_cv(scores = [0.96, 0.97, 1.98])

    # Output Errors

    def test_is_dict(self):
        assert isinstance(summary, dict)

    def test_output_length(self):
        assert len(summary) == 4

    def test_is_float(self):
        assert isinstance(summary['mean'], float)
        assert isinstance(summary['sd'], float)
        assert isinstance(summary['mode'], float)
        assert isinstance(summary['median'], float)

    def test_summary_cv(self):
        assert summary['mean'] == 0.9666667
        assert summary['sd'] == 0.01032796
        assert summary['mode'] == 0.97
        assert summary['median'] == 0.97
