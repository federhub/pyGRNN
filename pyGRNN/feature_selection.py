# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:03:06 2019

@author: famato1
"""

from .base import countX, combs, recursive_combination_excluder
from .GRNN import GRNN


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
import itertools 


class Isotropic_selector():
    """
        A class to perform supervised feature selection using Isotropic
        GRNN. It also allows an (unsupervised) analysis of the relatidness of 
        the input space.
    Parameters
    ----------
    bandwidth : str, default="rule-of-thumb"
        The strategy through which the kernel bandwidth is choosen. Can be one 
        between the Silverman's rule of thumb ('rule-of-thumb') and the grid search
        Cross Validation ('GSCV'). 
    """
    
    def __init__(self,bandwidth = 'rule-of-thumb'):
        self.bandwidth = bandwidth

    def isotropic_GSCV(X_tr, y_tr):
        '''Function to apply the Gridsearch CV to select the best bandwidth for 
            an isotropic GRNN.
        Parameters
        ----------
        X_tr : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the training features.
        y_tr : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets.'''
        IGRNN = GRNN()
        params_IGRNN = {'sigma' : list(np.arange(0.01, 1, 0.01)),
                        'calibration' : ['None']}
        grid_IGRNN = GridSearchCV(estimator=IGRNN,
                                  param_grid=params_IGRNN,
                                  scoring='neg_mean_squared_error',
                                  cv=5,
                                  verbose=1,
                                  n_jobs=-1)
        grid_IGRNN.fit(X_tr, y_tr.ravel()) 
        return grid_IGRNN.best_estimator_
            
    def es(self, X, y, feature_names):
        '''Exhaustive search evaluation of the importance of each feature 
           using IGRNN;can be used to detect relevant features in the input space.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the training features.
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets.
        feature_names : list
            A list containing the names of the input features.
        '''
        self.ex_search = {}
        self.X = X
        self.y = y
        self.best_inSpaceNames = []
        #s = 0.9 * (self.X.shape[0])**-(1/5) # Silverman's rule of thumb, for std=1 (z-scored data)
        var = list(np.arange(0, self.X.shape[1], 1))
        series =combs(var)#[1::]
        print('Exploring the ' +str(2**len(var)-1) + ' possible combination of features...')
        # Set counter
        i = 0
        # Run a model for each computation
        self.mse_cv = 999999 #initalising a very large int variable
        for fset in series:
            #print ('Exploring combination ' + str(i) + '...') 
            X_run= self.X[:, list(fset)] 
            X_train, X_test, y_train, y_test = train_test_split(preprocessing.StandardScaler().fit_transform(X_run),
                                                                y.reshape((-1, 1)),
                                                                test_size=0.25,
                                                                random_state = 42)
            # searching the optimal sigma, depending on the strategy defined
            if self.bandwidth == 'rule-of-thumb':
                s = 0.9 * (X_run.shape[0])**-(1/(4+ X_run.shape[1])) # Silverman's rule of thumb, for std=1 (z-scored data) see book Li, Racine page 66
                IGRNN = GRNN(sigma = s, calibration = 'None')
                best_model = IGRNN.fit(X_train, y_train.ravel())
            elif self.bandwidth == 'GSCV':
                best_model = Isotropic_selector.isotropic_GSCV(X_train, y_train.ravel())
            else: 
                raise ValueError('Bandwidth search strategy must be one between rule-of-thumb and GSCV!')
            # Run a GRNN with the selected bandwidth kernel 
            y_pred = best_model.predict(X_test)
            self.ex_search.update({fset :[MSE(y_test, y_pred)]})
            if (MSE(y_test, y_pred)) < self.mse_cv:
                self.mse_cv = MSE(y_test, y_pred)
                self.best_inSpaceIndex = fset 
                self.best_inSpace = X_run
                self.best_model = best_model
            else:
                pass
            #print('Combination ' + str(i) +' tested.')
            i += 1   
        [self.best_inSpaceNames.append(feature_names[n]) for n in self.best_inSpaceIndex]
        print('The best subset of features is ' + str(self.best_inSpaceNames))
        return self  
  
    def ffs(self, X, y, feature_names, stop_criterion = 'first_min'):
        '''Forward feature selctor for the evaluation of the importance of each
           feature using IGRNN;can be used to detect relevant features in the input space.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the training features.
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets.
        feature_names : list
            A list containing the names of the input features.
        stop_criterion: str, default="first_min"
            The stopping criterion of the algorithm. If 'first_min', the search
            will end as soon as adding another feature to the best subset causes 
            an increse of the MSE of the model. Differently, when 'full_search' is 
            selected, the entire forward feature selection will be performed. 
        '''
        y_ = y
        n_feat = np.arange(0, X.shape[1],1)
        self.mse_cv = 999999 #initalising a very large int variable
        mse_run_best = 999999 #initalising a very large int variable
        self.best_inSpaceNames = []
        best_comb = [{f} for f in n_feat] # initializing a list of sets for the definition of combinations
        for i in n_feat+1:
            if self.bandwidth == 'rule-of-thumb':
                s = 0.9 * (X.shape[0])**-(1/(4+ i)) # Silverman's rule of thumb, for std=1 (z-scored data) see book Li, Racine page 66    
            else:
                pass
            for fset in recursive_combination_excluder(n_feat, i, best_comb):
                X_ = X[:, list(fset)]
                X_train, X_test, y_train, y_test = train_test_split(preprocessing.StandardScaler().fit_transform(X_),
                                                                    y_.reshape((-1, 1)),
                                                                    test_size=0.25,
                                                                    random_state = 42)
                        # searching the optimal sigma, depending on the strategy defined
                if self.bandwidth == 'rule-of-thumb':
                    IGRNN = GRNN(sigma = s, calibration = 'None')
                    best_model = IGRNN.fit(X_train, y_train.ravel())
                elif self.bandwidth == 'GSCV':
                    best_model = Isotropic_selector.isotropic_GSCV(X_train, y_train.ravel())
                else: 
                    raise ValueError('Bandwidth search strategy must be one between rule-of-thumb and GSCV!')
                # Run a GRNN with the selected bandwidth kernel 
                y_pred = best_model.predict(X_test)
                if (MSE(y_test, y_pred)) < mse_run_best:
                    mse_run_best = MSE(y_test, y_pred)
                    best_comb_run = fset
                    X_run = X_
                else:
                    pass 
            if mse_run_best < self.mse_cv:
                self.best_inSpaceIndex = best_comb_run
                self.best_inSpace = X_run
                self.best_model = best_model
                self.mse_cv = mse_run_best   
                best_comb = [set(best_comb_run)]
            else:
                if stop_criterion == 'first_min':
                    break  
                else:
                    pass
        [self.best_inSpaceNames.append(feature_names[n]) for n in self.best_inSpaceIndex]
        return self, print('Best subset is: ' + str(self.best_inSpaceNames))
            
    def bfs(self, X, y, feature_names, stop_criterion = 'first_min'):
        '''Backward feature selctor for the evaluation of the importance of each
           feature using IGRNN;can be used to detect relevant features in the input space.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the training features.
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets.
        feature_names : list
            A list containing the names of the input features.
        stop_criterion: str, default="first_min"
            The stopping criterion of the algorithm. If 'first_min', the search
            will end as soon as adding another feature to the best subset causes 
            an increse of the MSE of the model. Differently, when 'full_search' is 
            selected, the entire forward feature selection will be performed.     
        '''
        y_ = y
        n_feat = -np.sort(-(np.arange(0, X.shape[1],1)))
        self.mse_cv = 999999 #initalising a very large int variable
        mse_run_best = 999999 #initalising a very large int variable
        self.best_inSpaceNames = []
        best_comb = ((f) for f in n_feat) # initializing a list of sets for the definition of combinations
        for i in n_feat+1:
            if self.bandwidth == 'rule-of-thumb':
                s = 0.9 * (X.shape[0])**-(1/(4+ i)) # Silverman's rule of thumb, for std=1 (z-scored data) see book Li, Racine page 66
            else:
                pass
            for fset in itertools.combinations(best_comb , i):            
                X_ = X[:, list(fset)]
                X_train, X_test, y_train, y_test = train_test_split(preprocessing.StandardScaler().fit_transform(X_),    
                                                                    y_.reshape((-1, 1)),
                                                                    test_size=0.25,
                                                                    random_state = 42)
                # searching the optimal sigma, depending on the strategy defined
                if self.bandwidth == 'rule-of-thumb':
                    IGRNN = GRNN(sigma = s, calibration = 'None')
                    best_model = IGRNN.fit(X_train, y_train.ravel())
                elif self.bandwidth == 'GSCV':
                    best_model = Isotropic_selector.isotropic_GSCV(X_train, y_train.ravel())
                else: 
                    raise ValueError('Bandwidth search strategy must be one between rule-of-thumb and GSCV!')
                # Run a GRNN with the selected bandwidth kernel 
                y_pred = best_model.predict(X_test)
                if (MSE(y_test, y_pred)) < mse_run_best:
                    mse_run_best = MSE(y_test, y_pred)
                    best_comb_run = fset
                    X_run = X_
                else:
                    pass   
            if mse_run_best < self.mse_cv:
                best_comb = ((f) for f in best_comb_run)
                self.best_inSpaceIndex = best_comb_run
                self.best_inSpace = X_run
                self.best_model = best_model
                self.mse_cv = mse_run_best  
                
            else:
                if stop_criterion == 'first_min':
                    break  
                else:
                    pass
        [self.best_inSpaceNames.append(feature_names[n]) for n in self.best_inSpaceIndex]
        return self, print('Best subset is: ' + str(self.best_inSpaceNames))

    
    def feat_selection(self, X, y, feature_names, strategy = 'ffs', stop_criterion = 'first_min'):
        ''' Feature selector based on IGRNN. For a given dataset, a complete
            exploration of the feature space is provided, giving information 
            about the relevance, irrelevance or the redundancy of the features.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the training features.
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets.
        feature_names : list
            A list containing the names of the input features.
        strategy : str, default = 'ffs'
            The search strategy for the feature selection. Can be forward ('ffs'),
            backward ('bfs') or exhaustive search ('es'). 
        stop_criterion: str, default="first_min"
            The stopping criterion of the algorithm. If 'first_min', the search
            will end as soon as adding another feature to the best subset causes 
            an increse of the MSE of the model. Differently, when 'full_search' is 
            selected, the entire forward feature selection will be performed.   
        '''
        in_feat = X
        output = y
        self.redundant = [] # we will store here the indexes of redundant features
        self.irrelevant = [] # we will store here the indexes of irrelevant features
        print('Searching relevant features...')
        if strategy is 'ffs':
            Isotropic_selector.ffs(self, in_feat, output, feature_names, stop_criterion = stop_criterion) # performing a forward feature selection to find relevant features
        elif strategy is 'bfs':
            Isotropic_selector.bfs(self, in_feat, output, feature_names, stop_criterion = stop_criterion) # performing backward feature selection to find relevant features
        else:
             Isotropic_selector.es(self, in_feat, output, feature_names) # performing an exhaustive search to find relevant features
        best__model_ = self.best_model 
        self.relevant = list(self.best_inSpaceIndex) # creating a list with the indexes of relevant features
        dataframe = pd.DataFrame(data=in_feat, columns=feature_names) # dataframe from the feature space
        for col in self.relevant:
            print('Searching the best subset to predict feature ' + str(col))
            feat = dataframe.drop(dataframe.columns[col], axis = 1).values
            target = dataframe.iloc[:,col].values
            # Getting the right names for the features
            feat_idx = [column for column in np.arange(0, in_feat.shape[1],1) if column != col] 
            feat_names = [feature_names[n] for n in feat_idx]
            # performing the search strategy
            if strategy is 'ffs':
                Isotropic_selector.ffs(self, feat, target, feat_names) # performing a forward feature selection to find relevant features
            elif strategy is 'bfs':
                Isotropic_selector.bfs(self, feat, target, feat_names) # performing backward feature selection to find relevant features
            else:
                Isotropic_selector.es(self, feat, target, feat_names) # performing an exhaustive search to find relevant features
            best_set_name = [feat_names[n] for n in  list(self.best_inSpaceIndex)] # getting the names of the best subset to predict the considered col 
            best_set_idx = [feature_names.index(m) for m in best_set_name] # getting the index of the best subset to predict the considered col 
            # usinf a (complicated) list cmprehension to list the feature which are
            # not in the best subset for the considered col
            not_relevant = [x for i,x in enumerate(list((range(0, in_feat.shape[1],1)))) if i not in best_set_idx] 
            print(not_relevant)
            for i in not_relevant: # if a feature cannot be used to predict a relevant feature, and it is not redundant for other relevant features, then it is irrelevant
                if i in self.relevant:
                    pass
                else:
                    if i in self.redundant:
                        pass
                    else:
                        if i not in self.irrelevant:
                            self.irrelevant.append(i)
                        else:
                            pass
            for j in best_set_idx: # if a feature can predict a relevant feature, it is redundant
                if j in self.relevant:
                    pass
                else:
                    if j in self.irrelevant:
                        self.irrelevant.remove(j)
                        if j not in self.redundant:
                            self.redundant.append(j)
                        else:
                            pass
                    else:
                        if j not in self.redundant:
                            self.redundant.append(j)
                        else:
                            pass   
        self.best_inSpaceIndex = self.relevant
        self.best_inSpace = X[:, self.relevant]
        self.best_model  = best__model_
        print('Research completed!')
        print('The relevant features are: ' + str([feature_names[n] for n in self.relevant]))    
        print('The redundant features are: ' + str([feature_names[n] for n in self.redundant]))   
        print('The irrelevant features are: ' + str([feature_names[n] for n in self.irrelevant]))   
        return self  
    
    def relatidness(self, X, feature_names, strategy = 'ffs'):
            '''Exhaustive search evaluation of the importance of each feature 
            in the input space using IGRNN; can be used to study the relatidness
            among the features in input or output space
            Parameters
            ----------
            X : array-like of shape = [n_samples, n_features]
                The input samples. Generally corresponds to the training features.
            feature_names : list
                A list containing the names of the input features.
            strategy : str, default = 'ffs'
                The search strategy for the feature selection. Can be forward ('ffs'),
                backward ('bfs') or exhaustive search ('es').              
            '''
            self.relatidness_ = {}
            dataframe = pd.DataFrame(data=X, columns=feature_names)
            #Defining a list of all the possible permutation of features
            for col in range(0,X.shape[1],1):
                print('Searching relatidness for feature ' + str(col))
                feat = dataframe.drop(dataframe.columns[col], axis = 1).values
                target = dataframe.iloc[:,col].values
                # Getting the right names for the features
                feat_idx = [column for column in np.arange(0, X.shape[1],1) if column != col] 
                feat_names = [feature_names[n] for n in feat_idx]
                # performing the search strategy
                if strategy is 'ffs':
                    Isotropic_selector.ffs(self, feat, target, feat_names) # performing a forward feature selection to find relevant features
                elif strategy is 'bfs':
                    Isotropic_selector.bfs(self, feat, target, feat_names) # performing backward feature selection to find relevant features
                else:
                    Isotropic_selector.es(self, feat, target, feat_names) # performing an exhaustive search to find relevant features
                self.relatidness_.update({col :[ [self.best_inSpaceNames]]})
            return self  
    
    def plot_(self, feature_names):
        '''A function to plot the relatidness matrix. Isotropic_selector.relatidness 
            has to be executed first.
         Parameters
            ----------
         feature_names : list
             A list containing the names of the input features.              
        '''
        if not self.relatidness_:
            raise ValueError("To plot results, Isotropic_selector.relatidness has to be executed first!")
        else:
            pass
        df = pd.DataFrame(columns = feature_names)
        feature = []
        best_set = []
        for k,v in self.relatidness_.items():
            feature.append(k)
            best_set.append(list(v[0][0]))
        col = 0
        for i in best_set: 
            presence = []
            for feat in feature_names:
                a = countX(i, feat)
                presence.append(a)
            df.iloc[:,col] = pd.Series(presence).values
            col += 1
        # plotting an heatmap with the best results
        plt.style.use('ggplot')
        with sns.axes_style("white"):
                f, ax = plt.subplots(figsize=(11, 9))
                ax.tick_params(labelsize = 14)
                ax.xaxis.set_ticks_position('top')
                cmap = sns.diverging_palette(220, 10, as_cmap=True)
                # Manually specify colorbar labelling after it's been generated
                p2 = sns.heatmap(df,cmap= cmap, square=True, linewidths=.5,
                                 yticklabels = feature_names, cbar_kws={"boundaries": (-1, 1, 2)})
                colorbar = ax.collections[0].colorbar
                colorbar.set_ticks([0, 1])
                colorbar.set_ticklabels(['Excluded', 'Included'])
        return p2                 
    
 
   
class Anisotropic_selector():
    """
        A class to perform supervised feature selection using Ansotropic
        GRNN. 
    """
    
    def bfe(self, X, y, feature_names, strategy = 'ffs',  stop_criterion = 'first_min', n_restarts_optimizer=0):
        '''Backward feature selctor for the evaluation of the importance of each
           feature using AGRNN;can be used to detect relevant features in the input space.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the training features.
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets.
        feature_names : list
            A list containing the names of the input features.
        stop_criterion: str, default="first_min"
            The stopping criterion of the algorithm. If 'first_min', the search
            will end as soon as adding another feature to the best subset causes 
            an increse of the MSE of the model. Differently, when 'full_search' is 
            selected, the entire forward feature selection will be performed.
        n_restarts_optimizer : int, default = 0
            The number of restarts of the optimizer for finding the kernel's
            parameters which maximize the cost function. The first run
            of the optimizer is performed from the kernel's initial parameters,
            the remaining ones (if any) from inital sigmas sampled log-uniform randomly
            from the space of allowed sigma-values. 
        '''
        self.mse_cv = 999999 #initalising a very large int variable
        y_ = y
        self.best_inSpaceNames = []
        self.best_inSpaceIndex = list(np.arange(0, X.shape[1],1))
                
        for maxiter in range(0, X.shape[1]):
            X_ = X[:, self.best_inSpaceIndex]
            X_train, X_test, y_train, y_test = train_test_split(X_,
                                                                y_.reshape((-1, 1)),
                                                                test_size=0.25,
                                                                random_state = 42)
            AGRNN = GRNN(calibration="warm_start", method="L-BFGS-B", n_restarts_optimizer=n_restarts_optimizer)
            best_model = AGRNN.fit(X_train, y_train.ravel())
            mse_run = MSE(y_test, best_model.predict(X_test))
            print('the current mse is '+str(mse_run))
            if mse_run < self.mse_cv:
                del self.best_inSpaceIndex[np.argmax(best_model.sigma)]
                self.best_inSpace = X_
                self.best_model = best_model
                self.mse_cv = mse_run        
            else :
                 if stop_criterion == 'first_min':
                    break  
                 else:
                    pass
        [self.best_inSpaceNames.append(feature_names[n]) for n in self.best_inSpaceIndex]
        return self, print('Best subset is: ' + str(self.best_inSpaceNames))
    
    def max_dist(self, X, y, feature_names, n_restarts_optimizer=0):
        '''Feature selctor for the evaluation of relevant feature using AGRNN.
           Can be used to detect relevant features in the input space. All the
           features which have bandwidth value lower than sqrt(n_feat) after 
           the model calibration are labeled as relevant.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the training features.
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets.
        feature_names : list
            A list containing the names of the input features.
        n_restarts_optimizer : int, default = 0
            The number of restarts of the optimizer for finding the kernel's
            parameters which maximize the cost function. The first run
            of the optimizer is performed from the kernel's initial parameters,
            the remaining ones (if any) from inital sigmas sampled log-uniform randomly
            from the space of allowed sigma-values. 
        '''
        AGRNN = GRNN(calibration="warm_start", method="L-BFGS-B", n_restarts_optimizer=n_restarts_optimizer)
        AGRNN.fit(X, y.ravel())
        self.best_inSpaceIndex = [i for i in range(len(AGRNN.sigma)) if AGRNN.sigma[i] < np.sqrt(len(AGRNN.sigma))]
        self.best_inSpaceNames = [feature_names[n] for n in self.best_inSpaceIndex]
        self.best_inSpace = X[:, self.best_inSpaceIndex]
        return self, print('Best subset is: ' + str(self.best_inSpaceNames))
