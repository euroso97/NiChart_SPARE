# """
# SPARE-AD Pipeline Module

# This module contains functions for training and inference of SPARE-AD models.
# """

# import pandas as pd
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score
# import joblib
# from typing import Tuple, Optional, Dict, Any

# # # Import common functions from util
# from ..util import (
#     get_svm_hyperparameter_grids,
# )

# from ..data_prep import (
#     validate_dataframe,
#     encode_feature_df,
#     preprocess_classification_data
# )


# # def preprocess_data(
# #     df: pd.DataFrame, 
# #     target_column: str,
# #     encode_categorical_features: bool = True,
# #     encode_categorical_target: bool = True,
# #     scale_features: bool = False,
# #     training: bool = True
# # ):
# #     if training == True:
# #         """Preprocess data for training: handle missing values and encode categorical targets."""
# #         # Remove rows with missing target values
# #         df = df.dropna(subset=[target_column])
        
# #         # Separate features and target
# #         X = df.drop(columns=[target_column])
# #         y = df[target_column]

# #         # Encode feature labels if they're not numeric and encoding is requested
# #         feature_encoder = None
# #         if encode_categorical_features:
# #             X, feature_encoder = encode_feature_df(X)
        
# #         # Encode target labels if they're not numeric and encoding is requested
# #         target_encoder = None
# #         if encode_categorical_target and y.dtype == 'object':
# #             target_encoder = LabelEncoder()
# #             y = target_encoder.fit_transform(y)
        
# #         # Scale features if requested
# #         scaler = None
# #         if scale_features:
# #             scaler = StandardScaler()
# #             X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
# #         return X, y, feature_encoder, target_encoder, scaler
    
# #     else:
# #         # Encode feature labels if they're not numeric and encoding is requested
# #         X = df
# #         feature_encoder = None
# #         if encode_categorical_features:
# #             X, feature_encoder = encode_feature_df(X)
# #         # Scale features if requested
# #         scaler = None
# #         if scale_features:
# #             scaler = StandardScaler()
# #             X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
# #         return X, feature_encoder, scaler

# def train_linearsvc_model():
#     print("Implement me")
#     model, feature_encoder, label_encoder, scaler = None, None, None, None
#     return model, feature_encoder, label_encoder, scaler


# # Accepts dataframe and target_column as input along with other parameters to perform an svc training
# def train_svc_model(
#     dataframe: pd.DataFrame,
#     target_column: str,
#     kernel: str = 'linear',
#     random_state: int = 42,
#     cv_fold: int = 5,
#     tune_hyperparameters: bool = False,
#     train_whole_set: bool = True,
#     **svc_params
# ):

#     # Input validation
#     print(f"Validating input...")
#     validate_dataframe(dataframe, target_column)
#     print(f"Success.")

#     # Preprocess the input df, split into X, y
#     print(f"Preprocessing the input...{dataframe.shape}")
#     X, y, feature_encoder, feature_scaler, target_encoder = preprocess_classification_data(dataframe, 
#                                                                                   target_column, 
#                                                                                   encode_categorical_features=True,
#                                                                                   scale_features=True,
#                                                                                   encode_categorical_target=True,
#                                                                                   for_training=True)
#     print(f"Input preprocessing completed.")
#     # Perform hyperparameter tuning when asked
#     if tune_hyperparameters:
#         param_grids = get_svm_hyperparameter_grids()['classification']
#         base_params = {'kernel': kernel, 'random_state': random_state}
#         base_params.update(svc_params)
      
#         # Get parameter grid for the specified kernel
#         param_grid = param_grids.get(kernel, {})
#         if param_grid:
#             # Remove any parameters that are already set in svc_params
#             for param in list(param_grid.keys()):
#                 if param in svc_params:
#                     del param_grid[param]
        
#         # Create base model
#         base_model = SVC(**base_params)
    
#         # Perform grid search with 5-fold CV
#         cv = RepeatedStratifiedKFold(n_splits=cv_fold, random_state=random_state)
#         grid_search = GridSearchCV(
#             base_model,
#             param_grid,
#             cv=cv,
#             scoring='balanced_accuracy',
#             n_jobs=-1,
#             verbose=3
#         )
        
#         grid_search.fit(X, y)
#         # model = grid_search.best_estimator_
    
#         # Get best parameters and CV score
#         cv_score = grid_search.best_score_
#         # Update the svc_params
#         svc_params = grid_search.best_params_

#         print(f"Best parameters: {svc_params}")
#         print(f"CV balanced accuracy: {cv_score:.3f}")

#     else:
#         # Use default parameters
#         svc_params.setdefault('random_state', random_state)
#         cv_score = None
#         # best_params = None

#     # Train model using the best parameter and whole set
#     if train_whole_set:
#         model = SVC(kernel=kernel, **svc_params)
#         model.fit(X, y)
        
#         return model, feature_encoder, feature_encoder, feature_scaler, target_encoder

#     else:
#         return grid_search.best_params_, feature_encoder, feature_encoder, feature_scaler, target_encoder