import pytest
import numpy as np
import pandas as pd
from CrossPy.CrossPy import crosspy

data = pd.read_csv("./test_data/test_data_short.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
y_list = list(y)

class Test_train_test_split():

    def test_type(self):
        with pytest.raise(TypeError):
            crosspy.train_test_split(X = X, y = y_list)

