import pytest
import numpy as np
import pandas as pd
#from CrossPy.CrossPy import crosspy
import CrossPy

data = pd.read_csv("./test_data/test_data_short.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
y_list = list(y)

class Test_train_test_split():

    def test_type(self):
        with pytest.raise(TypeError):
            crosspy.train_test_split(X = X, y = y_list)



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
