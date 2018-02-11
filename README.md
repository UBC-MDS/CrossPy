# CrossPy
The `CrossPy` package (short for _Cross_-validation in *Py*thon) is a package for performing cross-validation in Python.

### Contributors

* Nazli Ozum Kafaee / @nazliozum
* Daniel Raff / @raffrica
* Shun Chi / @ShunChi100

### Summary

Cross-validation is an important technique used in model selection and hyper-parameter optimization. Scores from cross-validation is a good estimation of test score of a predictive model in test data or new data as long as the IID assumption approximately holds in data. This package aims to provide a standardized pipeline for performing cross-validation for different modeling functions in Python. In addition, visualization of the cross-validation results is provided for users.  

### Functions

Three main functions in `CrossPy`:

- `split_data()`: This function partitions data into `k`-fold and returns the partitioned indices. A random shuffling option is provided. (`stratification` option for imbalanced representations will also be included if time allows.)

- `cross_validation()`: This function performs `k`-fold cross validation using the partitioned data and a selected model. It returns the scores of each validation.

- `plot()`: This function visualizes the cross-validation scores against the tuning of hyper-parameters. For many hyper-parameters, it outputs a grid of plots with one plot for one hyper-parameter.

### Similar packages

The [`scikit-learn`](http://scikit-learn.org/stable/) library in Python implements the first two function we propose in [`sklearn.model_selection.train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) and [`sklearn.cross_validation`](http://scikit-learn.org/stable/modules/cross_validation.html). However, we have realized that although plotting cross-validation scores against hyper-parameter values is something we often do, there are no existing functions to implement such plot directly. Therefore, we consider a `plot()` function of `CrossPy` as an addition to the functions offered by `sklearn` for cross-validation.


### License
[MIT License](https://github.com/UBC-MDS/CrossPy/blob/master/LICENSE)

### Contributing
This is an open source project. So feedback, suggestions and contributions are very welcome. For feedback and suggestions, please open an issue in this repo. If you are willing to contribute this package, please refer [Contributing](https://github.com/UBC-MDS/CrossPy/blob/master/CONTRIBUTING.md) guidelines for details.
