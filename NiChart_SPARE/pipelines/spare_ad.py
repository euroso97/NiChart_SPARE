"""
SPARE-AD Pipeline Module

This module contains functions for training and inference of SPARE-AD models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from typing import Tuple, Optional, Dict, Any

# # Import common functions from util
from ..util import (
    validate_dataframe, 
#     save_model, 
#     load_model, 
#     predict_model, 
#     get_feature_importance,
#     create_training_info
)

from ..data_prep import (
    preprocess_data,
    get_svm_hyperparameter_grids
)

# Accepts dataframe and target_column as input along with other parameters to perform an svc training
def train_svc_model(
    dataframe: pd.DataFrame,
    target_column: str,
    kernel: str = 'linear',
    random_state: int = 42,
    cv_fold: int = 5,
    tune_hyperparameters: bool = False,
    train_whole_set: bool = True,
    **svc_params
    #return_best_params: bool = False,
    ):

    # Input validation
    validate_dataframe(dataframe, target_column)

    # Preprocess the input df, split into X, y
    X, y, feature_encoder, label_encoder, scaler = preprocess_data(dataframe, 
                                                                   target_column, 
                                                                   encode_categorical_features=True,
                                                                   encode_categorical_target=True,
                                                                   scale_features=True
                                                                   )
    # Perform hyperparameter tuning when asked
    if tune_hyperparameters:
        param_grids = get_svm_hyperparameter_grids()['classification']
        base_params = {'kernel': kernel, 'random_state': random_state}
        base_params.update(svc_params)
      
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
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_fold,
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_state
        )
        
        grid_search.fit(X, y)
        # model = grid_search.best_estimator_
    
        # Get best parameters and CV score
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"CV accuracy: {cv_score:.3f}")

        # Update the svc_params
        svc_params = grid_search.best_params_
    else:
        # Use default parameters
        svc_params.setdefault('random_state', random_state)
        cv_score = None
        best_params = None

    # Train model using the best parameter and whole set
    if train_whole_set:
        model = SVC(kernel=kernel, **svc_params)
        model.fit(X, y)
        
        return model, feature_encoder, label_encoder, scaler

    else:
        return grid_search.best_params_, feature_encoder, label_encoder, scaler