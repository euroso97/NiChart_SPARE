"""
Sklearn SVM specific functions
"""
import sys
import joblib
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR

import numpy as np
# import pandas as pd

from .util import expspace, is_regression_model
from .data_prep import load_csv_data, validate_dataframe, preprocess_classification_data, preprocess_regression_data
from .pipelines import spare_svm_classification, spare_svm_regression#, spare_ad, spare_ba, spare_ht


def get_pipeline_module(spare_type):
    """Get the appropriate pipeline module based on SPARE type"""
    spare_type = spare_type.upper()
    pipeline_map = {
        'CL': spare_svm_classification,
        'RG': spare_svm_regression,
        # 'AD': spare_ad,
        # 'BA': spare_ba,
        # 'HT': spare_ht,
    }
    if spare_type not in pipeline_map:
        raise ValueError(f"Unsupported SVM SPARE type: {spare_type}")
    
    return pipeline_map[spare_type]


# def create_svm_training_info(
#     model = None,
#     test_metrics: dict = None,
#     cv_scores: Optional[dict] = None, # {'fold_0': {'metric_0':0.x, 'metric_1':0.x, ...}, 'fold_1': {'metric_0':0.x, ...}, ...}
#     best_params: Optional[dict] = None,
#     df: pd.DataFrame = None,
#     X_features: pd.DataFrame = None,
#     kernel: str = None,
#     tune_hyperparameters: bool = None,
#     final_model: svm = None
# ) -> dict:
#     """Create training information dictionary."""
#     training_info = {
#         'n_samples': len(df),
#         'n_features': X_features.shape[1],
#         'kernel': kernel,
#         'feature_names': list(X_features.columns),
#         'hyperparameter_tuned': tune_hyperparameters,
#         'final_model': final_model
#     }
    
#     # Add test metrics
#     training_info.update(test_metrics)
    
#     # Add CV score and best parameters if available
#     if cv_scores is not None:
#         training_info['cv_scores'] = cv_scores
#     if best_params is not None:
#         training_info['best_params'] = best_params
    
#     return training_info

def save_svm_model(
    model, 
    scaler: StandardScaler, 
    training_info: dict, 
    filepath: str
) -> None:
    """Save the trained model and components to a file."""
    model_data = {
        'model': model,
        'scaler': scaler,
        'training_info': training_info
    }
    joblib.dump(model_data, filepath)


def load_svm_model(filepath: str) -> Tuple[Any, StandardScaler, dict, Optional[LabelEncoder]]:
    """Load a trained model and components from a file."""
    model_data = joblib.load(filepath)
    
    model = model_data['model']
    scaler = model_data['scaler']
    training_info = model_data['training_info']
    # label_encoder = model_data.get('label_encoder', None)
    
    return model, scaler, training_info #, label_encoder


def correct_svr_bias(svr_model, X_train, y_train):
    """
    Corrects the regularization bias of a trained SVR model by adjusting its intercept.

    The regularization in SVR can introduce a systematic bias in the predictions.
    This function corrects this bias by calculating the mean of the residuals
    on the training data and adding it to the model's intercept.

    Args:
        svr_model: A trained scikit-learn SVR model.
        X_train: The training data (features).
        y_train: The training data (target values).

    Returns:
        A new SVR model object with the corrected intercept.
    """
    # Create a new SVR model to avoid modifying the original one
    corrected_model = SVR(
        C=svr_model.C,
        epsilon=svr_model.epsilon,
        kernel=svr_model.kernel,
        degree=svr_model.degree,
        gamma=svr_model.gamma,
        coef0=svr_model.coef0,
    )

    # Fit the new model to have the same support vectors and coefficients
    corrected_model.fit(X_train, y_train)

    # Predict on the training data using the original model
    y_pred_train = svr_model.predict(X_train)

    # Calculate the residuals (difference between actual and predicted values)
    residuals = y_train - y_pred_train

    # Calculate the mean of the residuals, which is the bias
    bias = np.mean(residuals)

    # Correct the intercept of the new model by adding the bias
    corrected_model.intercept_ += bias

    return corrected_model


def train_svm_model(input_file, 
					model_path, 
					spare_type, 
					target_column, 
					kernel, 
					tune_hyperparameters, 
                    cv_fold,
                    class_balancing,
					train_whole_set, 
					drop_columns=None):
    
    """Train model using the pipeline functions"""
    # Load data
    print("Loading training data...")
    df = load_csv_data(input_file, drop_columns=drop_columns)
    
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
    
    # Get pipeline module
    pipeline_module = get_pipeline_module(spare_type)


    
    if spare_type in ['RG','BA']:
        model, scaler = None
        # Regression model (BA - Brain Age)

        # Input validation
        print(f"Validating input...")
        validate_dataframe(df, target_column)
        print(f"Success.")
        
        # Preprocess data (no label encoding for regression)
        print(f"Preprocessing the input...{df.shape}")
        X, y, feature_encoder, feature_scaler, target_scaler = preprocess_regression_data( 
            df, 
            target_column = target_column, 
            encode_categorical_features=True,
            scale_features=True,
            for_training=True)
        print(f"Input preprocessing completed.")

        if kernel=='linear':
            print("Training model with LinearSVR...")
            # model = pipeline_module.train_linearsvr_model(
            #     X,
            #     y,
            #     kernel=kernel,
            #     tune_hyperparameters=tune_hyperparameters,
            #     cv_fold=cv_fold,
            #     get_cv_scores=True,
            #     train_whole_set=train_whole_set
            # )
        else:
            # Standard training
            print(f"Training model with default SVR with {kernel} kernel...")
            model = pipeline_module.train_svr_model(
                X,
                y,
                kernel=kernel,
                tune_hyperparameters=tune_hyperparameters,
                cv_fold=cv_fold,
                get_cv_scores=True,
                train_whole_set=train_whole_set
            )
        
        if model != None & scaler != None & model_path != None:
            # Save model
            save_svm_model(model, scaler, {}, model_path)
            print(f"Model saved to: {model_path}")
    
    elif spare_type in ['CL','AD']:
        
        # Input validation
        print(f"Validating input...")
        validate_dataframe(df, target_column)
        print(f"Success.")

        # Preprocess the input df, split into X, y
        print(f"Preprocessing the input...{df.shape}")
        X, y, feature_encoder, feature_scaler, target_encoder = preprocess_classification_data(
            df, 
            target_column = target_column, 
            encode_categorical_features=True,
            encode_categorical_target=True,
            scale_features=True,
            for_training=True)
        print(f"Input preprocessing completed.")

        # Training
        if kernel.lower()=='linear':
            print("Training model with LinearSVR...")
            model = pipeline_module.train_linearsvc_model(
                X,
                y,
                kernel=kernel,
                tune_hyperparameters=tune_hyperparameters,
                cv_fold=cv_fold,
                class_balancing=class_balancing,
                get_cv_scores=True,
                train_whole_set=train_whole_set
                )
        elif kernel.lower() in ['poly', 'rbf', 'sigmoid']:
            model = pipeline_module.train_svc_model(
                X,
                y,
                kernel=kernel,
                tune_hyperparameters=tune_hyperparameters,
                cv_fold=cv_fold,
                class_balancing=class_balancing,
                get_cv_scores=True,
                train_whole_set=train_whole_set
                )
        else:
            print(f"Unsupported SVM kernel entry. Please select among: linear, poly, rbf, sigmoid.")
        
        if model != None and feature_scaler != None and model_path != None:
            # Save model
            save_svm_model(model, feature_scaler, {}, model_path)
            print(f"Model saved to: {model_path}")

    else:
        print(f"{spare_type} is not supported.")
        sys.exit(1)


def predict_svm_model(input_file, 
                      model_path, 
                      output_file, 
                      spare_type, 
                      drop_columns=None):
    """Make predictions using trained model"""
    
    # Load data
    print("Loading prediction data...")
    df = load_csv_data(input_file, drop_columns=drop_columns)
    
    # Get pipeline module
    pipeline_module = get_pipeline_module(spare_type)
    
    # Load model
    print("Loading trained model...")
    
    if is_regression_model(spare_type):
        # Regression model
        model, scaler, info = pipeline_module.load_model(model_path) # TBF
        predictions = pipeline_module.predict_svr(model, scaler, df)
    else:
        # Classification model
        model, scaler, encoder, info = pipeline_module.load_model(model_path) # TBF
        predictions = pipeline_module.predict_svc(model, scaler, df, encoder)
    
    # Create output dataframe
    output_df = df.copy()
    output_df['predicted_target'] = predictions
    
    # Save predictions
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")