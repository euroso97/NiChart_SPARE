# SPARE module to train a misc biomarker using undefined set

"""
SPARE-AD Pipeline Module

This module contains functions for training and inference of SPARE-AD models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
# import joblib
# from typing import Tuple, Optional, Dict, Any

# # Import common functions from util
from ..util import (
    get_svm_hyperparameter_grids
)

from ..data_analysis import report_classification_metrics


def train_linearsvc_model(
    X,
    y,
    kernel: str = 'linear',
    tune_hyperparameters: bool = False,
    cv_fold: int = 5,
    class_balancing: bool = True,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    random_state: int = 42,
    **svc_params):

    print("Implement me")
    model, feature_encoder, label_encoder, scaler = None, None, None, None
    return model, feature_encoder, label_encoder, scaler


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
    random_state: int = 42,
    **svc_params
):
    # Initialize base parameters
    base_params = {'kernel': kernel, 'random_state': random_state}
    base_params.update(svc_params)  # overwrite base_params
    
    # Enable class_weight='balanced' if class_balancing parameter is passed and True
    if class_balancing:
        base_params.update({'class_weight':'balanced'})

    # Perform hyperparameter tuning when asked
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        param_grids = get_svm_hyperparameter_grids()['classification']
      
        # Get parameter grid for the specified kernel
        param_grid = param_grids.get(kernel, {})
        if param_grid:
            # Remove any parameters that are already set in svc_params
            for param in list(param_grid.keys()):
                if param in svc_params:
                    del param_grid[param]
        
        # Create base model
        base_model = SVC(**base_params)
    
        # Perform grid search with 5-fold CV
        cv = RepeatedStratifiedKFold(n_splits=cv_fold,
                                     n_repeats=10, 
                                     random_state=random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=3
        )
        
        grid_search.fit(X, y)
    
        # Get best parameters and CV score & Update the svc_params
        cv_score = grid_search.best_score_
        svc_params = grid_search.best_params_

        print(f"Best parameters: {svc_params}")
        print(f"CV balanced accuracy: {cv_score:.3f}")

    else:
        print(f"Hyperparameter selection skipped...")
        # Use default parameters
        svc_params.setdefault('random_state', random_state)
        cv_score = None
        # best_params = None

    # Perform another CV using the best parameter if get_cv_score parameter is True
    if get_cv_scores:
        print(f"Initiating {cv_fold}-fold CV")
        cv_scores = {}
        cv = RepeatedStratifiedKFold(n_splits=cv_fold, 
                                     n_repeats=10, 
                                     random_state=random_state)
        
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            
            # Train model with current parameters
            model = SVC(kernel=kernel, **svc_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            # Get validation metrics
            cv_metric = report_classification_metrics(y_test, y_pred)
        cv_score["Fold_%d"%i] = cv_metric

    # Train model using the best parameter and whole set
    if train_whole_set:
        model = SVC(kernel=kernel, **svc_params)
        model.fit(X, y)
        return model
    
    else:
        if tune_hyperparameters:
            return grid_search.best_estimator_
        else:
            return None