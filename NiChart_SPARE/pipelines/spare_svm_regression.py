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
    get_svm_hyperparameter_grids, 
)

from ..data_prep import (
    validate_dataframe,
    encode_feature_df,
)

# def train_linearsvr_model(
#     dataframe: pd.DataFrame,
#     target_column: str,
#     tune_hyperparameters: bool = False,
#     cv_fold: int = 5,
#     get_cv_scores: bool = True,
#     train_whole_set: bool = True,
#     **svr_params
# ):
#     """Train an SVR model to predict the target column from a dataframe."""
    
#   return


def train_svr_model(
    X,
    y,
    kernel: str = 'linear',
    tune_hyperparameters: bool = False,
    cv_fold: int = 5,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    random_state: int = 42,
    **svc_params
):
    """Train an SVR model to predict the target column from a dataframe."""
    
    # Initialize base parameters
    base_params = {'kernel': kernel, 'random_state': random_state}
    base_params.update(svc_params)  # overwrite base_params
        
    # Train SVR model
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        param_grids = get_svm_hyperparameter_grids()['regression']
        
        # Get parameter grid for the specified kernel
        param_grid = param_grids.get(kernel, {})
        if param_grid:
            # Remove any parameters that are already set in svc_params
            for param in list(param_grid.keys()):
                if param in svc_params:
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
        print(f"Hyperparameter selection with {cv_fold} fold CV completed.")
        
        # Get best parameters and CV score & Update the svc_params
        cv_score = grid_search.best_score_
        svr_params = grid_search.best_params_
        
        print(f"Best parameters: {svr_params}")
        print(f"CV balanced accuracy: {cv_score:.3f}")

    else:
        print(f"Hyperparameter selection skipped...")
        # Use default parameters
        svc_params.setdefault('random_state', random_state)
        cv_score = None
    
    if get_cv_scores:
        print(f"Initiating {cv_fold}-fold CV")
        cv_scores = []
        cv = RepeatedKFold(n_splits=cv_fold, n_repeats=10, random_state=random_state)
        
        for i, (train_index, test_index) in enumerate(cv.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train model with current parameters
            model = SVR(kernel=kernel, **svc_params)
            model.fit(X_train, y_train)
            
            # Predict and calculate accuracy
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            cv_scores.append(score)
            
            print(f"Fold {i+1}: R2 = {score:.3f}")
        
        print(f"Mean CV R2: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Train model using the best parameter and whole set
    if train_whole_set:
        model = SVR(kernel=kernel, **svc_params)
        model.fit(X, y)
        return model
    
    else:
        if tune_hyperparameters:
            return grid_search.best_estimator_
        else:
            return None