# pyGRNN
Python implementation of General Regression Neural Network (GRNN, also known as Nadaraya-Watson Estimator). A Feature Selection module based on GRNN is also provided.

## Getting Started with GRNN

GRNN is an adaptation in terms of neural network of the Nadaraya-Watson estimator, with which the general regression of a scalar on a vector independent variable is computed as a locally weighted average with a kernel as a weighting function. The main advantage of this algorithm is that its calibration only requires the definition of a proper bandwidth for the kernel estimation. Hence, GRNN is faster than other feedforward artificial neural network algorithms.

The traditional GRNN architecture is based on the use of one unique value of the bandwidth for all the features. This Isotropic structure of the network (IGRNN) can be used as a wrapper for feature selection. This approach permits a complete description of the input space, identifying relevant, irrelevant and redundant features. Specifically, redundancy and irrelevancy are associated to the identification of relatedness, i.e. the non-linear predictability of an input variable using the other features of the input space. 
Anisotropic (or Adaptive) GRNN (AGRNN) are an evolution of GRNN in which different values are given to the bandwidth corresponding to each feature. A proper calibration of the bandwidths will scale the input features depending on their explanatory power. Specifically, a large smoothing parameter will give rise to a lower discriminative power of the associated feature, and vice versa. Hence, AGRNN can be considered as an embedded feature selection method in which the bandwidth values of the kernel express a measure of the relevancy of the features.

For more insights on GRNN, check [these slides] (https://github.com/federhub/pyGRNN/blob/master/Feature_Selection_with_GRNN.pdf) or try the [step-by-step tutorial in jupyter notebook!] (https://github.com/federhub/pyGRNN/blob/master/Tutorial/Tutorial_PyGRNN.ipynb)

## Install

```sh
$ pip install pyGRNN
```

## Examples

pyGRNN can be used to perform Isotropic and Anisotropic General Regression Neural Networks:

```sh
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from PyGRNN import GRNN
# Loading the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(preprocessing.minmax_scale(X),
                                                    preprocessing.minmax_scale(y.reshape((-1, 1))),
                                                    test_size=0.25)

# Example 1: use Isotropic GRNN with a Grid Search Cross validation to select the optimal bandwidth
IGRNN = GRNN.GRNN()
params_IGRNN = {'kernel':["RBF"],
                'sigma' : list(np.arange(0.1, 4, 0.01)),
                'calibration' : ['None']
                 }
grid_IGRNN = GridSearchCV(estimator=IGRNN,
                          param_grid=params_IGRNN,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=1
                          )
grid_IGRNN.fit(X_train, y_train.ravel())
best_model = grid_IGRNN.best_estimator_
y_pred = best_model.predict(X_test)
mse_IGRNN = MSE(y_test, y_pred)

# Example 2: use Anisotropic GRNN with Limited-Memory BFGS algorithm to select the optimal bandwidths
AGRNN = GRNN.GRNN(calibration="gradient_search")
AGRNN.fit(X_train, y_train.ravel())
sigma=AGRNN.sigma 
y_pred = AGRNN.predict(X_test)
mse_AGRNN = MSE(y_test, y_pred)
```

The package can also be used to perform feature selection:

```sh
from PyGRNN import feature_selection as FS
# Loading the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
featnames = diabetes.feature_names
# Example 1: use Isotropic Selector to explore data dependencies in the input 
# space by analyzing the relatedness between features 
IsotropicSelector = FS.Isotropic_selector(bandwidth = 'rule-of-thumb')
IsotropicSelector.relatidness(X, feature_names = featnames)
IsotropicSelector.plot_(feature_names = featnames)
# Example 2: use Isotropic Selector to perform an exhaustive search; a rule-of-thumb
# is used to select the optimal bandwidth for each subset of features
IsotropicSelector = FS.Isotropic_selector(bandwidth = 'rule-of-thumb')
IsotropicSelector.es(X, y.ravel(), feature_names=featnames)
# Example 2: use Isotropic Selector to perform a complete analysis of the input
# space, recongising relevant, redundant, irrelevant features
IsotropicSelector = FS.Isotropic_selector(bandwidth = 'rule-of-thumb')
IsotropicSelector.feat_selection(X, y.ravel(), feature_names=featnames, strategy ='es')
```

## Authors

* **Federico Amato** - *Postdoctoral Research Fellow,* - [GeoKDD lab, University of Lausanne](https://wp.unil.ch/geokdd/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
