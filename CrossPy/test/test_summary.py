import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

import pytest
import pandas as pd
from CrossPy.CrossPy import summary_cv

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
    assert len(summary_cv(gen_summary())) == 3

def test_mean_is_float():
    assert isinstance(summary_cv(gen_summary())['mean'], float)

def test_median_is_float():
    assert isinstance(summary_cv(gen_summary())['median'], float)

def test_sd_is_float():
    assert isinstance(summary_cv(gen_summary())['sd'], float)

def test_summary_cv_mean():
    assert summary_cv(gen_summary())['mean'] == 0.967

def test_summary_cv_median():
    assert summary_cv(gen_summary())['median'] == 0.97

def test_summary_cv_sd():
    assert summary_cv(gen_summary())['sd'] == 0.009
