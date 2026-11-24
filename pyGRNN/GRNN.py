# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:30:37 2019

@author: famato1
"""
import numpy as np
from operator import itemgetter
import warnings
from scipy import optimize
from sklearn.metrics import mean_squared_error as MSE
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.gaussian_process import kernels
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

class GRNN(BaseEstimator, RegressorMixin):
    """A General Regression Neural Network, based on the Nadaraya-Watson estimator.
    Parameters
    ----------
    kernel : str, default="RBF"
        Kernel function to be casted in the regression.
        The radial basis function (RBF) is used by default: K(x,x')=exp(-|x-x'|^2/(2 sigma^2))

    sigma : float, array, default=0.4
        Bandwidth standard deviation parameter for the Kernel. All Kernels are casted from the
        sklearn.metrics.pairwise library. Check its documentation for further
        info and to find a list of all the built-in Kernels.

    n_splits : int, default=5
        Number of folds used in the K-Folds CV definition of the sigma.
        Must be at least 2.

    calibration : str, default="warm_start"
        Type of calibration of the sigma.
        'gradient_search' minimizes the loss function by applying the scipy.optimize.minimize function.
        Gradient search can be used for isotropic Kernels (sigma = int, float),
        or on anisotropic Kernel (sigma = list).
        'warm_start' is used when sigma is a single scalar. Gradient search is applied
        to find the best sigma value for an isotropic Kernel (all sigmas are the same). The optimized
        parameter is then used as a starting point to search the optimal solution for an
        anisotropic Kernel (having one sigma per feature).

    method : str, default="L-BFGS-B"
        Type of solver for the gradient search (used to find the local minimum of the cost function).
        Other choises (such as the CG based on the Polak and
        Ribiere algorithm) are discussed on the help of the scipy function.

    bnds : list, default=(0, None)
        (min, max) pairs for each element in x, defining the bounds on that parameter.
        Use None or +-inf for one of min or max when there is no bound in that direction.

    n_restarts_optimizer : int, default = 0
        The number of restarts of the optimizer for finding the kernel's
        parameters which minimize the cost function. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from inital sigmas sampled log-uniform randomly
        from the space of allowed sigma-values.

    seed : int, default = 42
        Random state used to initialize random generators.
    
    
    Notes
    -----
    shortcut: `fit()` writes the calibrated bandwidth back into `self.sigma`
    (the same attribute used as the constructor parameter), so `sklearn.base.clone()`
    of an already-fitted instance carries the calibrated value over instead of
    resetting it to the constructor default. Construct a fresh GRNN(...) rather
    than cloning a fitted one if you need the original default back. Upgrade
    path: split into `sigma` (param) and `sigma_` (fitted), a breaking change
    for every example in this README/docstring.

    This Python code was developed and used for the following papers:
    F. Amato, F. Guignard, P. Jacquet, M. Kanveski. Exploration of data dependencies
    and feature selection using General Regression Neural Networks.

   
   References
    ----------
    F. Amato, F. Guignard, P. Jacquet, M. Kanveski. Exploration of data dependencies
    and feature selection using General Regression Neural Networks.

    D.F. Specht. A general regression neural network. IEEE transactions on neural 
    networks 2.6 (1991): 568-576.


    Examples
    --------
    import numpy as np
    from sklearn import datasets
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import  GridSearchCV
    from sklearn.metrics import mean_squared_error as MSE

    from pyGRNN import GRNN

    # Loading the diabetes dataset
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(preprocessing.minmax_scale(X),
                                                        preprocessing.minmax_scale(y.reshape((-1, 1))),
                                                        test_size=0.25)
    # Example 1: use Isotropic GRNN with a Grid Search Cross validation to select the optimal bandwidth
    IGRNN = GRNN()
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
    AGRNN = GRNN(calibration="gradient_search")
    AGRNN.fit(X_train, y_train.ravel())
    sigma=AGRNN.sigma 
    y_pred = AGRNN.predict(X_test)
    mse_AGRNN = MSE(y_test, y_pred)
    """

    def __init__(self, kernel='RBF', sigma=0.4, n_splits=5, calibration='warm_start', method='L-BFGS-B', bnds=(0, None), n_restarts_optimizer=0, seed = 42):
        self.kernel = kernel
        self.sigma = sigma
        self.n_splits = n_splits
        self.calibration = calibration
        self.method = method
        self.bnds = bnds
        self.n_restarts_optimizer = n_restarts_optimizer
        self.seed = seed
        
    def fit(self, X, y):
        """Fit the model.  
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples. Generally corresponds to the training features
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets
        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        # Local working copies: `self.bnds`/`self.sigma` (constructor params) are
        # never overwritten here, so fit() can be called again on the same instance.
        bnds_ = self.bnds
        sigma0 = self.sigma

        def cost(sigma_):
            '''Cost function to be minimized. It computes the cross validation
                error for a given sigma vector.'''
            kf = KFold(n_splits= self.n_splits, random_state=self.seed, shuffle=True)
            kf.get_n_splits(self.X_)
            cv_err = []
            # Zero-densities are possible with very high/low distances; scoped to
            # this block so it doesn't silence unrelated NumPy warnings for the
            # rest of the process (np.seterr used to do exactly that, permanently).
            with np.errstate(divide='ignore', invalid='ignore'):
                for train_index, validate_index in kf.split(self.X_):
                    X_tr, X_val = self.X_[train_index], self.X_[validate_index]
                    y_tr, y_val = self.y_[train_index], self.y_[validate_index]
                    Kernel_def_= getattr(kernels, self.kernel)(length_scale=sigma_)
                    K_ = Kernel_def_(X_tr, X_val)
                    K_ = np.nan_to_num(K_)
                    psum_ = K_.sum(axis=0).T # Cumulate denominator of the Nadaraya-Watson estimator
                    psum_ = np.nan_to_num(psum_)
                    y_pred_ = (np.dot(y_tr.T, K_) / psum_)
                    y_pred_ = np.nan_to_num(y_pred_)
                    cv_err.append(MSE(y_val, y_pred_.T))
            return np.mean(cv_err, axis=0) ## Mean error over the k splits
        
        def optimization(x0_):
            '''A function to find the optimal values of sigma (i.e. the values 
               minimizing the cost) given an inital guess x0.'''
            opt = optimize.minimize(cost, x0_, method=self.method, bounds=bnds_)
            if opt['success'] is True:
                opt_sigma = opt['x']
                opt_cv_error = opt['fun']
            else:
                warnings.warn('Optimization may have failed. Consider changing optimization solver for the gradient search')
                opt_sigma = np.full(len(self.X_[0]), np.nan)
                opt_cv_error = np.inf
                pass
            return [opt_sigma, opt_cv_error]

        def pick_best(optima):
            '''Given a list of [sigma, cv_error] optimization results, return the
            sigma and cv_error of the run with the lowest cv_error.'''
            cost_values = list(map(itemgetter(1), optima))
            best_sigma = optima[np.argmin(cost_values)][0]
            return best_sigma, np.min(cost_values)

        def calibrate_sigma(x0):
            '''A function to find the values of sigma minimizing the CV-MSE, given
            a starting guess x0 (either user-defined or measured with warm start).
            The optimization is based on scipy.optimize.minimize.'''
            if self.n_restarts_optimizer > 0:
                #First optimize starting from theta specified in kernel
                optima = [optimization(x0)] 
                # # Additional runs are performed from log-uniform chosen initial bandwidths
                r_s = np.random.RandomState(self.seed)
                for iteration in range(self.n_restarts_optimizer): 
                    x0_iter = np.full(len(self.X_[0]), np.around(r_s.uniform(0,1), decimals=3))
                    optima.append(optimization(x0_iter))             
            elif self.n_restarts_optimizer == 0:        
                optima = [optimization(x0)]            
            else:
                raise ValueError('n_restarts_optimizer must be a positive int!')

            # Select sigma from the run minimizing cost
            best_sigma, self.cv_error = pick_best(optima)
            return best_sigma


        if self.calibration == 'warm_start':
            print('Executing warm start...')
            bnds_ = (self.bnds,)
            x0 = np.asarray(sigma0)
            optima = [optimization(x0)]
            warm_sigma, _ = pick_best(optima)
            print('Warm start concluded. The optimum isotropic sigma is ' + str(warm_sigma))
            x0 = np.full(len(self.X_[0]), np.around(warm_sigma, decimals=3))
            bnds_ = (self.bnds,)*len(self.X_[0])
            #print ('Executing gradient search...')
            self.sigma = calibrate_sigma(x0)
            print('Gradient search concluded. The optimum sigma is ' + str(self.sigma))
        elif self.calibration == 'gradient_search':
            #print ('Executing gradient search...')
            x0 = np.full(len(self.X_[0]), sigma0)
            bnds_ = (self.bnds,)*len(self.X_[0])
            self.sigma = calibrate_sigma(x0)
            #print('Gradient search concluded. The optimum sigma is ' + str(self.sigma))
        else:
            pass
                   
        self.is_fitted_ = True
        # Return the regressor
        return self
     
    def predict(self, X):
        """Predict target values for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the testing features
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted target value.
        """
        
         # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # Input validation
        X = check_array(X)
        
        Kernel_def= getattr(kernels, self.kernel)(length_scale=self.sigma)
        K = Kernel_def(self.X_, X)
        # If the distances are very high/low, zero-densities must be prevented:
        K = np.nan_to_num(K)
        psum = K.sum(axis=0).T # Cumulate denominator of the Nadaraya-Watson estimator
        psum = np.nan_to_num(psum)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.dot(self.y_.T, K) / psum)
