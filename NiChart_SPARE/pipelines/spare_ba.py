"""
SPARE-BA Pipeline Module

This module contains functions for training and inference of SPARE-BA models.
Brain Age is a continuous variable, so this uses regression models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Tuple, Optional, Dict, Any

# Import common functions from util
from ..util import (
    validate_dataframe, 
    preprocess_data,
    save_model, 
    load_model, 
    predict_model, 
    get_feature_importance,
    create_training_info
)

def train_svr_model(
    dataframe: pd.DataFrame,
    target_column: str,
    kernel: str = 'rbf',
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = False,
    return_best_params: bool = False,
    **svr_params
) -> Tuple[SVR, StandardScaler, dict]:
    """Train an SVR model to predict the target column from a dataframe."""
    
    # Input validation
    validate_dataframe(dataframe, target_column)
    
    # Preprocess data (no label encoding for regression)
    X, y, _ = preprocess_data(dataframe, target_column, encode_categorical=False)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get hyperparameter grids
    param_grids = get_hyperparameter_grids()['regression']
    
    # Train SVR model
    if tune_hyperparameters:
        # Use GridSearchCV for hyperparameter tuning
        base_params = {'kernel': kernel, 'random_state': random_state}
        base_params.update(svr_params)
        
        # Get parameter grid for the specified kernel
        param_grid = param_grids.get(kernel, {})
        if param_grid:
            # Remove any parameters that are already set in svr_params
            for param in list(param_grid.keys()):
                if param in svr_params:
                    del param_grid[param]
        
        # Create base model
        base_model = SVR(**base_params)
        
        # Perform grid search with 5-fold CV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=random_state
        )
        
        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_
        
        # Get best parameters and CV score
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        
        print(f"Best parameters: {best_params}")
        print(f"CV R² score: {cv_score:.3f}")
        
    else:
        # Use default parameters
        svr_params.setdefault('random_state', random_state)
        model = SVR(kernel=kernel, **svr_params)
        model.fit(X_train_scaled, y_train)
        cv_score = None
        best_params = None
    
    # Calculate test metrics
    y_test_pred = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Create training info
    test_metrics = {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }
    training_info = create_training_info(
        model, test_metrics, cv_score, best_params, 
        dataframe, X, kernel, tune_hyperparameters
    )
    
    return model, scaler, training_info

def train_final_model(
    dataframe: pd.DataFrame,
    target_column: str,
    best_params: Dict[str, Any],
    kernel: str = 'rbf',
    random_state: int = 42
    ) -> Tuple[SVR, StandardScaler, dict]:
    """Train a final SVR model using the best hyperparameters on the entire dataset."""
    
    # Input validation
    validate_dataframe(dataframe, target_column)
    
    if not best_params:
        raise ValueError("best_params cannot be empty")
    
    # Preprocess data (no label encoding for regression)
    X, y, _ = preprocess_data(dataframe, target_column, encode_categorical=False)
    
    # Scale the features using the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model with best parameters
    model_params = {
        'kernel': kernel,
        'random_state': random_state,
        **best_params
    }
    
    model = SVR(**model_params)
    
    # Train on the entire dataset
    model.fit(X_scaled, y)
    
    # Create training info
    training_info = create_training_info(
        model, {}, None, best_params, 
        dataframe, X, kernel, False, final_model=True
    )
    
    print(f"Final model trained on {len(dataframe)} samples")
    print(f"Using best parameters: {best_params}")
    
    return model, scaler, training_info

# Use common functions from util.py
save_model = save_model
load_model = load_model
predict_svr = predict_model
get_feature_importance = get_feature_importance
