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
    """
    Train an SVR model to predict the target column from a dataframe.
    """
    
    # Input validation
    if dataframe.empty:
        raise ValueError("Dataframe is empty")
    
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Remove rows with missing target values
    dataframe = dataframe.dropna(subset=[target_column])
    
    # Separate features and target
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    # Fill missing values in features with median
    X = X.fillna(X.median())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define hyperparameter grids for different kernels
    param_grids = {
        'linear': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2]
        },
        'rbf': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        },
        'poly': {
            'C': [0.1, 1, 10],
            'degree': [2, 3],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1]
        },
        'sigmoid': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'epsilon': [0.01, 0.1]
        }
    }
    
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
    
    # Training info
    training_info = {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_score': cv_score,
        'best_params': best_params,
        'n_samples': len(dataframe),
        'n_features': X.shape[1],
        'kernel': kernel,
        'feature_names': list(X.columns),
        'tuned': tune_hyperparameters
    }
    
    return model, scaler, training_info

def train_final_model(
    dataframe: pd.DataFrame,
    target_column: str,
    best_params: Dict[str, Any],
    kernel: str = 'rbf',
    random_state: int = 42
) -> Tuple[SVR, StandardScaler, dict]:
    """
    Train a final SVR model using the best hyperparameters on the entire dataset.
    """
    
    # Input validation
    if dataframe.empty:
        raise ValueError("Dataframe is empty")
    
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    if not best_params:
        raise ValueError("best_params cannot be empty")
    
    # Remove rows with missing target values
    dataframe = dataframe.dropna(subset=[target_column])
    
    # Separate features and target
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    # Fill missing values in features with median
    X = X.fillna(X.median())
    
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
    
    # Training info
    training_info = {
        'n_samples': len(dataframe),
        'n_features': X.shape[1],
        'kernel': kernel,
        'feature_names': list(X.columns),
        'best_params': best_params,
        'final_model': True
    }
    
    print(f"Final model trained on {len(dataframe)} samples")
    print(f"Using best parameters: {best_params}")
    
    return model, scaler, training_info

def save_model(model, scaler, training_info, filepath):
    """Save the trained model and components"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'training_info': training_info
    }
    joblib.dump(model_data, filepath)

def load_model(filepath):
    """Load a trained model and components"""
    model_data = joblib.load(filepath)
    return (
        model_data['model'],
        model_data['scaler'],
        model_data['training_info']
    )

def predict_svr(model, scaler, dataframe):
    """Make predictions using a trained SVR model"""
    # Handle missing values
    dataframe = dataframe.fillna(dataframe.median())
    
    # Scale features
    X_scaled = scaler.transform(dataframe)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions

def get_feature_importance(model, feature_names):
    """Get feature importance from SVR model (for linear kernel only)"""
    if model.kernel != 'linear':
        print("Feature importance is only meaningful for linear kernels")
        return pd.DataFrame()
    
    # Get feature importance from linear SVR
    importance = np.abs(model.coef_[0])
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df
