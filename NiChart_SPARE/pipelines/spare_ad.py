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
    """
    Train an SVC model to predict the target column from a dataframe.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing features and target column
    target_column : str
        Name of the target column to predict
    kernel : str, default='rbf'
        SVC kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random state for reproducibility
    tune_hyperparameters : bool, default=False
        Whether to perform hyperparameter tuning with 5-fold CV
    return_best_params : bool, default=False
        Whether to return best parameters for final training
    **svc_params : dict
        Additional parameters to pass to SVC constructor
        
    Returns
    -------
    tuple
        (trained_svc_model, scaler, label_encoder, training_info)
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
    
    # Encode target labels if they're not numeric
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define hyperparameter grids for different kernels
    param_grids = {
        'linear': {
            'C': [0.1, 1, 10, 100],
            'class_weight': [None, 'balanced']
        },
        'rbf': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'class_weight': [None, 'balanced']
        },
        'poly': {
            'C': [0.1, 1, 10],
            'degree': [2, 3],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced']
        },
        'sigmoid': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'class_weight': [None, 'balanced']
        }
    }
    
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
    
    # Training info
    training_info = {
        'test_accuracy': test_accuracy,
        'cv_score': cv_score,
        'best_params': best_params,
        'n_samples': len(dataframe),
        'n_features': X.shape[1],
        'kernel': kernel,
        'feature_names': list(X.columns),
        'tuned': tune_hyperparameters
    }
    
    return model, scaler, label_encoder, training_info

def train_final_model(
    dataframe: pd.DataFrame,
    target_column: str,
    best_params: Dict[str, Any],
    kernel: str = 'rbf',
    random_state: int = 42
) -> Tuple[SVC, StandardScaler, Optional[LabelEncoder], dict]:
    """
    Train a final SVC model using the best hyperparameters on the entire dataset.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing features and target column
    target_column : str
        Name of the target column to predict
    best_params : dict
        Best hyperparameters found during tuning
    kernel : str, default='rbf'
        SVC kernel type
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    tuple
        (final_trained_model, scaler, label_encoder, training_info)
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
    
    # Encode target labels if they're not numeric
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
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
    
    return model, scaler, label_encoder, training_info

def save_model(model, scaler, label_encoder, training_info, filepath):
    """Save the trained model and components"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'training_info': training_info
    }
    joblib.dump(model_data, filepath)

def load_model(filepath):
    """Load a trained model and components"""
    model_data = joblib.load(filepath)
    return (
        model_data['model'],
        model_data['scaler'],
        model_data['label_encoder'],
        model_data['training_info']
    )

def predict_svc(model, scaler, dataframe, label_encoder=None):
    """Make predictions using a trained SVC model"""
    # Handle missing values
    dataframe = dataframe.fillna(dataframe.median())
    
    # Scale features
    X_scaled = scaler.transform(dataframe)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Decode predictions if label encoder was used
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)
    
    return predictions

def get_feature_importance(model, feature_names):
    """Get feature importance from SVC model (for linear kernel only)"""
    if model.kernel != 'linear':
        print("Feature importance is only meaningful for linear kernels")
        return pd.DataFrame()
    
    # Get feature importance from linear SVC
    importance = np.abs(model.coef_[0])
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df
