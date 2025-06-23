"""
SPARE-HT Pipeline Module

This module contains functions for training and inference of SPARE-HT models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from typing import Tuple, Optional, Dict, Any

# Import common functions from util
from ..util import (
    validate_dataframe, preprocess_data, get_hyperparameter_grids,
    save_model, load_model, predict_model, get_feature_importance,
    create_training_info
)

def train_svc_model(
    dataframe: pd.DataFrame,
    target_column: str,
    kernel: str = 'rbf',
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = False,
    return_best_params: bool = False,
    **svc_params
) -> Tuple[SVC, StandardScaler, Optional[LabelEncoder], dict]:
    """Train an SVC model to predict the target column from a dataframe."""
    
    # Input validation
    validate_dataframe(dataframe, target_column)
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(dataframe, target_column, encode_categorical=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get hyperparameter grids
    param_grids = get_hyperparameter_grids()['classification']
    
    # Train SVC model
    if tune_hyperparameters:
        # Use GridSearchCV for hyperparameter tuning
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
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_state
        )
        
        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_
        
        # Get best parameters and CV score
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"CV accuracy: {cv_score:.3f}")
        
    else:
        # Use default parameters
        svc_params.setdefault('random_state', random_state)
        model = SVC(kernel=kernel, **svc_params)
        model.fit(X_train_scaled, y_train)
        cv_score = None
        best_params = None
    
    # Calculate test accuracy
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Create training info
    test_metrics = {'test_accuracy': test_accuracy}
    training_info = create_training_info(
        model, test_metrics, cv_score, best_params, 
        dataframe, X, kernel, tune_hyperparameters
    )
    
    return model, scaler, label_encoder, training_info

def train_final_model(
    dataframe: pd.DataFrame,
    target_column: str,
    best_params: Dict[str, Any],
    kernel: str = 'rbf',
    random_state: int = 42
) -> Tuple[SVC, StandardScaler, Optional[LabelEncoder], dict]:
    """Train a final SVC model using the best hyperparameters on the entire dataset."""
    
    # Input validation
    validate_dataframe(dataframe, target_column)
    
    if not best_params:
        raise ValueError("best_params cannot be empty")
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(dataframe, target_column, encode_categorical=True)
    
    # Scale the features using the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model with best parameters
    model_params = {
        'kernel': kernel,
        'random_state': random_state,
        **best_params
    }
    
    model = SVC(**model_params)
    
    # Train on the entire dataset
    model.fit(X_scaled, y)
    
    # Create training info
    training_info = create_training_info(
        model, {}, None, best_params, 
        dataframe, X, kernel, False, final_model=True
    )
    
    print(f"Final model trained on {len(dataframe)} samples")
    print(f"Using best parameters: {best_params}")
    
    return model, scaler, label_encoder, training_info

# Use common functions from util.py
save_model = save_model
load_model = load_model
predict_svc = predict_model
get_feature_importance = get_feature_importance
