"""
SPARE-BA Pipeline Module

This module contains functions for training and inference of SPARE-BA models.
Brain Age is a continuous variable, so this uses regression models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Tuple, Optional, Dict, Any

# Import common functions from util
from ..util import (
    validate_dataframe, 
    save_model, 
)

from ..data_prep import (
    encode_feature_df,
    get_svm_hyperparameter_grids
)

def preprocess_data(
    df: pd.DataFrame, 
    target_column: str,
    encode_categorical_features: bool = True,
    scale_features: bool = False
):
    """Preprocess data for training: handle missing values and encode categorical targets."""
    # Remove rows with missing target values
    df = df.dropna(subset=[target_column])
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode feature labels if they're not numeric and encoding is requested
    feature_encoder = None
    if encode_categorical_features:
        X, feature_encoder = encode_feature_df(X)
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    return X, y, feature_encoder, scaler

def train_linearsvr_model(
    dataframe: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
    cv_fold: int = 5,
    tune_hyperparameters: bool = False,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    **svr_params
):
    """Train an SVR model to predict the target column from a dataframe."""
    
    svr_params = {"loss":"squared_epsilon_insensitive",
                  "dual":False}
    
    # Input validation
    print(f"Validating input...")
    validate_dataframe(dataframe, target_column)
    print(f"Success.")
    
    # Preprocess data (no label encoding for regression)
    print(f"Preprocessing the input...{dataframe.shape}")
    X, y, _, scaler = preprocess_data(dataframe, 
                                   target_column, 
                                   scale_features=True)
    print(f"Input preprocessing completed.")
    
    # Perform hyperparameter tuning when asked
    if tune_hyperparameters:
        param_grids = get_svm_hyperparameter_grids()['regression']
        print(f"Hyperparameter selectio initated...")
        base_params = {'random_state': random_state}
        base_params.update(svr_params)
      
        # # Get parameter grid for the specified kernel
        param_grid = param_grids.get('linear', {})
        
        if param_grid:
            # Remove any parameters that are already set in svc_params
            for param in list(param_grid.keys()):
                if param in svr_params:
                    del param_grid[param]
        
        # Create base model
        base_model = LinearSVR(**base_params)
        
        # Perform grid search with 5-fold CV
        cv = RepeatedKFold(n_splits=cv_fold)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=3
        )
        
        grid_search.fit(X, y)
        # Get best parameters and CV score
        #best_params = grid_search.best_params_
        print(f"Hyperparameter selection with {cv_fold} fold CV completed.")
        cv_score = grid_search.best_score_
        # Update the svc_params
        svr_params = grid_search.best_params_
        
        print(f"Best parameters: {svr_params}")
        print(f"CV R2: {cv_score:.3f}")

    else:
        print(f"Hyperparameter selection skipped...")
        cv_score = None
    
    if get_cv_scores:
        print(f"Initiating {cv_fold}-fold CV")
        cv = RepeatedKFold(n_splits=cv_fold, random_state=2025)
        model = LinearSVR(**svr_params)
        scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        cv_results = cross_validate(model, 
                                    X, 
                                    y, 
                                    cv=cv_fold, 
                                    scoring=scoring_metrics, 
                                    return_train_score=True,
                                    verbose=1)
        cv_results_df = pd.DataFrame(cv_results)
        # Make the MSE scores positive for easier interpretation
        cv_results_df['test_mae'] = -cv_results_df['test_neg_mean_absolute_error']
        cv_results_df['train_mae'] = -cv_results_df['train_neg_mean_absolute_error']
        cv_results_df['test_mse'] = -cv_results_df['test_neg_mean_squared_error']
        cv_results_df['train_mse'] = -cv_results_df['train_neg_mean_squared_error']

        print("5-Fold Cross-Validation Results using cross_validate:")
        print("-----------------------------------------------------")
        print("Full results dictionary as DataFrame:")
        print(cv_results_df[['fit_time', 'test_mae', 'train_mae', 'test_r2', 'train_r2', 'test_mse', 'train_mse']])
        print("\n" + "="*50 + "\n")

        print(f"Summary of {cv_fold}-fold CV:")
        print("--------------------------------")
        print(f"Mean Fit Time: {cv_results_df['fit_time'].mean():.4f}s")

        # Summary for R-squared
        mean_r2 = cv_results_df['test_r2'].mean()
        std_r2 = cv_results_df['test_r2'].std()
        print(f"\nMean R-squared: {mean_r2:.4f} (std: {std_r2:.4f})")
        # Summary for MAE
        mean_mae = cv_results_df['test_mae'].mean()
        std_mae = cv_results_df['test_mae'].std()
        print(f"Mean MAE: {mean_mae:.4f} (std: {std_mae:.4f})")
        # Summary for Mean Squared Error
        mean_mse = cv_results_df['test_mse'].mean()
        std_mse = cv_results_df['test_mse'].std()
        print(f"Mean MSE: {mean_mse:.4f} (std: {std_mse:.4f})")
        # Check for overfitting by comparing train and test scores
        mean_train_r2 = cv_results_df['train_r2'].mean()
        mean_test_r2 = cv_results_df['test_r2'].mean()
        print(f"\nAverage Train R-squared: {mean_train_r2:.4f}")
        print(f"Average Test R-squared:  {mean_test_r2:.4f}")
        if mean_train_r2 > mean_test_r2 + (2*std_r2):
            print("Note: The model may be overfitting, as the training score is significantly higher than the test score.")


    # Train model using the best parameter and whole set
    if not train_whole_set:
        return grid_search.best_params_, scaler
    
    else:
        model = LinearSVR(**svr_params)
        model.fit(X, y)
        return model, scaler


def train_svr_model(
    dataframe: pd.DataFrame,
    target_column: str,
    kernel: str = 'linear',
    cv_fold: int = 5,
    tune_hyperparameters: bool = False,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    **svr_params
):
    """Train an SVR model to predict the target column from a dataframe."""
    
    # Input validation
    print(f"Validating input...")
    validate_dataframe(dataframe, target_column)
    print(f"Success.")
    
    # Preprocess data (no label encoding for regression)
    print(f"Preprocessing the input...{dataframe.shape}")
    X, y, scaler = preprocess_data(dataframe, 
                                   target_column, 
                                   scale_features=True)
    print(f"Input preprocessing completed.")
    
    # Get hyperparameter grids
    param_grids = get_svm_hyperparameter_grids()['regression']
    
    # Train SVR model
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        # Use GridSearchCV for hyperparameter tuning
        base_params = {'kernel': kernel}
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
        cv = RepeatedKFold(n_splits=cv_fold)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=3
        )
        
        grid_search.fit(X, y)
        # Get best parameters and CV score
        #best_params = grid_search.best_params_
        print(f"Hyperparameter selection with {cv_fold} fold CV completed.")
        cv_score = grid_search.best_score_
        # Update the svc_params
        svr_params = grid_search.best_params_
        
        print(f"Best parameters: {svr_params}")
        print(f"CV balanced accuracy: {cv_score:.3f}")

    else:
        print(f"Hyperparameter selection skipped...")
        cv_score = None
    
    if get_cv_scores:
        print(f"Initiating 5-fold CV")
        cv = RepeatedKFold(n_splits=cv_fold)
        model = SVR(kernel=kernel, **svr_params)
        scoring_metrics = ['r2', 'neg_mean_squared_error']
        cv_results = cross_validate(model, 
                                    X, 
                                    y, 
                                    cv=cv_fold, 
                                    scoring=scoring_metrics, 
                                    return_train_score=True,
                                    verbose=1)
        cv_results_df = pd.DataFrame(cv_results)
        # Make the MSE scores positive for easier interpretation
        cv_results_df['test_mse'] = -cv_results_df['test_neg_mean_squared_error']
        cv_results_df['train_mse'] = -cv_results_df['train_neg_mean_squared_error']
        print("5-Fold Cross-Validation Results using cross_validate:")
        print("-----------------------------------------------------")
        print("Full results dictionary as DataFrame:")
        print(cv_results_df[['fit_time', 'test_r2', 'train_r2', 'test_mse', 'train_mse']])
        print("\n" + "="*50 + "\n")

        print("Summary of Test Set Performance:")
        print("--------------------------------")
        print(f"Mean Fit Time: {cv_results_df['fit_time'].mean():.4f}s")

        # Summary for R-squared
        mean_r2 = cv_results_df['test_r2'].mean()
        std_r2 = cv_results_df['test_r2'].std()
        print(f"\nMean R-squared: {mean_r2:.4f} (std: {std_r2:.4f})")
        # Summary for Mean Squared Error
        mean_mse = cv_results_df['test_mse'].mean()
        std_mse = cv_results_df['test_mse'].std()
        print(f"Mean MSE: {mean_mse:.4f} (std: {std_mse:.4f})")
        # Check for overfitting by comparing train and test scores
        mean_train_r2 = cv_results_df['train_r2'].mean()
        mean_test_r2 = cv_results_df['test_r2'].mean()
        print(f"\nAverage Train R-squared: {mean_train_r2:.4f}")
        print(f"Average Test R-squared:  {mean_test_r2:.4f}")
        if mean_train_r2 > mean_test_r2 + (2*std_r2):
            print("Note: The model may be overfitting, as the training score is significantly higher than the test score.")


    # Train model using the best parameter and whole set
    if not train_whole_set:
        return grid_search.best_params_, scaler
    else:
        model = SVR(kernel=kernel, **svr_params)
        model.fit(X, y)
        return model, scaler
    