# SPARE module to train a misc biomarker using undefined set

"""
SPARE-AD Pipeline Module

This module contains functions for training and inference of SPARE-AD models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from ..data_analysis import (
	report_classification_metrics
)

from ..util import (
    get_hyperparameter_tuning
)

from ..svm import (
    get_svm_hyperparameter_grids
)

def train_linearsvc_model(
    X,
    y,
    tune_hyperparameters: bool = False,
    cv_fold: int = 5,
    class_balancing: bool = True,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    random_state: int = 42, # for replication
    verbose: int = 1,
    **svc_params
    ):

        # Items to return
    model = None
    grid_search = None
    cv_scores = None
    best_cv_model = None
    best_cv_score = 0
    
    # Initialize base parameters
    base_params = {'random_state': random_state,
                   'verbose' : verbose > 1}
    # Overwrite base parameters with svc_params
    base_params.update(svc_params)
    
    # Enable class_weight='balanced' if class_balancing parameter is passed and True
    if class_balancing:
        base_params.update({'class_weight':'balanced'})
    
    # Perform hyperparameter tuning when asked
    hyperparameter_tuning={}
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        param_grids = get_svm_hyperparameter_grids()['classification']

        # Create base model
        base_model = LinearSVC(**base_params)
    
        # Perform grid search with 5-fold CV
        cv = RepeatedStratifiedKFold(n_splits=cv_fold,
                                     n_repeats=5, 
                                     random_state=random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grids['linear'],
            cv=cv,
            scoring='balanced_accuracy' if class_balancing == True else 'accuracy',
            n_jobs=-1,
            verbose=verbose
        )
        
        grid_search.fit(X, y)
    
        # Get best parameters and CV score & Update the svc_params
        # cv_score = grid_search.best_score_
        base_params.update(grid_search.best_params_)

        print(f"Best parameters: {base_params}")
        print(f"Best CV {grid_search.scorer_}: {grid_search.best_score_:.3f}")

        hyperparameter_tuning = get_hyperparameter_tuning(grid_search, base_params, param_grids)

    else:
        print(f"Hyperparameter selection skipped...")
        # Use default parameters
        svc_params.setdefault('random_state', random_state)

    # Perform another CV using the best parameter if get_cv_score parameter is True
    cv_scores = {}
    if get_cv_scores:
        print(f"Initiating {cv_fold}-fold CV")
        
        cv = RepeatedStratifiedKFold(n_splits=cv_fold, 
                                     n_repeats=5, 
                                     random_state=random_state)
        
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            
            # Train model with current parameters
            model = LinearSVC(**base_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            # Get validation metrics
            cv_metric = report_classification_metrics(y_test, y_pred)
            print(f"Iteration {i+1} Repeat {(i+1)//cv_fold} Fold {i % cv.n_repeats} metrics: {cv_metric}")
            # Save the scores
            cv_scores["Fold_%d" % (i % cv.n_repeats)] = cv_metric
            # Update the best performing model based off of ROC-AUC
            if cv_metric['ROC-AUC'] > best_cv_score:
                best_cv_model = model
                best_cv_score = cv_metric['ROC-AUC']
            

    # Train model using the best parameter and whole set
    if train_whole_set:
        print("Training the wholeset.")
        model = LinearSVC(**base_params)
        model.fit(X, y)
        #return model, cv_scores
    
    else:
        if tune_hyperparameters:
            model = grid_search.best_estimator_
        elif get_cv_scores:
            model = best_cv_model
    
    # Return model and the CV scores
    return model, hyperparameter_tuning, cv_scores


# Accepts dataframe and target_column as input along with other parameters to perform an svc training
def train_svc_model(
    X,
    y,
    kernel: str = 'linear',
    tune_hyperparameters: bool = False,
    cv_fold: int = 5,
    class_balancing: bool = True,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    random_state: int = 42, # for replication
    verbose: int = 1,
    **svc_params
    ):
    # Items to return
    model = None
    grid_search = None
    cv_scores = None
    best_cv_model = None
    best_cv_score = 0
    
    # Initialize base parameters
    base_params = {'kernel': kernel, 
                   'random_state': random_state,
                   'verbose' : verbose > 1}
    # Overwrite base parameters with svc_params
    base_params.update(svc_params)
    
    # Enable class_weight='balanced' if class_balancing parameter is passed and True
    if class_balancing:
        base_params.update({'class_weight':'balanced'})
    
    # Perform hyperparameter tuning when asked
    hyperparameter_tuning={}
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        param_grids = get_svm_hyperparameter_grids()['classification']
             
        # Create base model
        base_model = SVC(**base_params)
    
        # Perform grid search with 5-fold CV
        cv = RepeatedStratifiedKFold(n_splits=cv_fold,
                                     n_repeats=5, 
                                     random_state=random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grids[kernel],
            cv=cv,
            scoring='balanced_accuracy' if class_balancing == True else 'accuracy',
            n_jobs=-1,
            verbose=verbose
        )
        
        grid_search.fit(X, y)
    
        # Get best parameters and CV score & Update the svc_params
        # cv_score = grid_search.best_score_
        base_params.update(grid_search.best_params_)

        print(f"Best parameters: {base_params}")
        print(f"Best CV {grid_search.scorer_}: {grid_search.best_score_:.3f}")

        hyperparameter_tuning = get_hyperparameter_tuning(grid_search, base_params, param_grid)

    else:
        print(f"Hyperparameter selection skipped...")
        # Use default parameters
        svc_params.setdefault('random_state', random_state)

    # Perform another CV using the best parameter if get_cv_score parameter is True
    cv_scores = {}
    if get_cv_scores:
        print(f"Initiating {cv_fold}-fold CV")
        
        cv = RepeatedStratifiedKFold(n_splits=cv_fold, 
                                     n_repeats=5, 
                                     random_state=random_state)
        
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            
            # Train model with current parameters
            model = SVC(**base_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            # Get validation metrics
            cv_metric = report_classification_metrics(y_test, y_pred)
            print(f"Iteration {i+1} Repeat {(i+1)//cv_fold} Fold {i % cv.n_repeats} metrics: {cv_metric}")
            # Save the scores
            cv_scores["Fold_%d" % (i % cv.n_repeats)] = cv_metric
            # Update the best performing model based off of ROC-AUC
            if cv_metric['ROC-AUC'] > best_cv_score:
                best_cv_model = model
                best_cv_score = cv_metric['ROC-AUC']
            

    # Train model using the best parameter and whole set
    if train_whole_set:
        print("Training the wholeset.")
        model = SVC(**base_params)
        model.fit(X, y)
        #return model, cv_scores
    
    else:
        if tune_hyperparameters:
            model = grid_search.best_estimator_
        elif get_cv_scores:
            model = best_cv_model
    
    # Return model and the CV scores
    return model, hyperparameter_tuning, cv_scores


# def predict_svc(df, model, meta_data, preprocessor, verbose=0):
#     # preprocess the data
#     df_set = df[meta_data['training_data_description']['feature_names']]

#     if verbose > 1:
#         print(f"Features used: {df_set.columns}")
    
#     if preprocessor['feature_encoder'] != None:
        
#     if preprocessor['feature_scaler'] != None:

#     if preprocessor['target_encoder'] != None:

#     if preprocessor['target_scaler'] != None: