# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:30:37 2019

@author: famato1
"""
import numpy as np
from scipy import optimize
from sklearn.metrics import mean_squared_error as MSE
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.gaussian_process import kernels
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

class GRNN(BaseEstimator, RegressorMixin):
    """A General Regression Neural Network, based on the Nadaraya-Watson estimator.
    Parameters:
    ----------
    kernel : str, default="rbf"
        Kernel function to be casted in the regression.
        The radial basis function (rbf) is used by default: K(x,x')=exp(-|x-x'|^2/(2 sigma^2))
        
    sigma : float, array, default=None
        Bandwidth standard deviation parameter for the Kernel. All Kernels are casted from the 
        sklearn.metrics.pairwise library. Check its documentation for further 
        info and to find a list of all the built-in Kernels.
    
    n_splits : int, default 10
        Number of folds used in the K-Folds CV definition of the sigma. 
        Must be at least 2.
    
    calibration : str, default=None
        Type of calibration of the sigma. 
        'gradient_search' minimizes the loss function by applying the scipy.optimize.minimize function. 
        Gradient search can be used for isotropic Kernels (sigma = int, float),
        or on anisotropic Kernel (sigma = list).
        'warm_start' is used when sigma is a single scalar. Gradient search is applied 
        to find the best sigma value for an isotropic Kernel (all sigmas are the same). The optimized 
        parameter is then used as a starting point to search the optimal solution for an 
        anisotropic Kernel (having one sigma per feature).
    
    method: str, default=Nelder-Mead
        Type of solver for the gradient search (used to find the local minimum of the cost function). 
        The default solver used is the Nelder-Mead. 
        Other choises (such as the CG based on the Polak and 
        Ribiere algorithm) are discussed on the help of the scipy function.
        
    """

    def __init__(self, kernel='RBF', sigma=0.4, n_splits=5, calibration='warm_start', method='Nelder-Mead', bnds=(0, None)):
        self.kernel = kernel
        self.sigma = sigma
        self.n_splits = n_splits
        self.calibration = calibration
        self.method = method
        self.bnds = bnds
        
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
        bounds = self.bnds
         
        if self.calibration is 'warm_start':
            self.bnds = (bounds,)
            GRNN.warm_start(self)           
            self.bnds = (bounds,)*len(self.X_[0])
            GRNN.calibrate_sigma(self)
        elif self.calibration is 'gradient_search':
            self.sigma = np.full(len(self.X_[0]), self.sigma)
            self.bnds = (bounds,)*len(self.X_[0])
            GRNN.calibrate_sigma(self)
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
        return np.nan_to_num((np.dot(self.y_.T, K) / psum))
    
    def calibrate_sigma(self):
        np.seterr(divide='ignore', invalid='ignore')
      
        def cost(sigma_):
            Kernel_def_= getattr(kernels, self.kernel)(length_scale=sigma_)
            K_ = Kernel_def_(X_tr, X_val)
            # If the distances are very high/low, zero-densities must be prevented:
            K_ = np.nan_to_num(K_)
            psum_ = K_.sum(axis=0).T # Cumulate denominator of the Nadaraya-Watson estimator
            psum_ = np.nan_to_num(psum_)
            y_pred_ = (np.dot(y_tr.T, K_) / psum_)
            y_pred_ = np.nan_to_num(y_pred_)
            return MSE(y_val, y_pred_)
        print ('Executing gradient search')
        features = self.X_
        targets = self.y_
        kf = KFold(n_splits= self.n_splits, random_state=42)
        kf.get_n_splits(features)
        
        sigma_kf = []
        opt_fun = []
        for train_index, validate_index in kf.split(features):
            X_tr, X_val = features[train_index], features[validate_index]
            y_tr, y_val = targets[train_index], targets[validate_index]
            x0 = np.asarray(self.sigma)
            opt = optimize.minimize(cost, x0, method=self.method, bounds=self.bnds)
            if opt['success'] is True:
                sigma_kf.append(opt['x'])
                opt_fun.append(opt['fun'])
            else:
                pass           
        if not sigma_kf:
            self.sigma = np.full(len(self.X_[0]), np.nan)
            #raise ValueError("Optimization failed!")
            #pass
        else:
            self.sigma = np.mean(sigma_kf, axis=0) ## Use this line to set sigma to the mean value obtained over the k-fold CV
            #self.sigma= sigma_kf[opt_fun.index(min(opt_fun))] # Use this line to set sigma to the value minimizing error over the K-fold CV
        return self, print('Gradient search concluded. The optimum sigma is ' + str(self.sigma))
    
    def warm_start(self):
        print ('Executing warm start')
        if self.calibration == 'warm_start':
            GRNN.calibrate_sigma(self)
            self.sigma = np.full(len(self.X_[0]), self.sigma)
        return self, print('Warm start concluded. The optimum isotropic sigma is ' + str(self.sigma))