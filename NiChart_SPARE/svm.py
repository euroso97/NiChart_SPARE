"""
Sklearn SVM specific functions
"""
import sys
import joblib
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm

import pandas as pd

from .util import expspace, is_regression_model
from .data_prep import load_csv_data
from .pipelines import spare_svm_classification, spare_svm_regression, spare_ad, spare_ba, spare_ht


def get_pipeline_module(spare_type):
    """Get the appropriate pipeline module based on SPARE type"""
    spare_type = spare_type.upper()
    pipeline_map = {
        'CL': spare_svm_classification,
        'RG': spare_svm_regression,
        'AD': spare_ad,
        'BA': spare_ba,
        'HT': spare_ht,
    }
    
    if spare_type not in pipeline_map:
        raise ValueError(f"Unsupported SVM SPARE type: {spare_type}")
    
    return pipeline_map[spare_type]


def get_svm_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """Get hyperparameter grids for different kernels and model types."""
    classification_grids = {
        'linear': {
            "C": expspace([-9, 5])
        },
        'rbf': {
            "C": expspace([-9, 5]),
            "gamma": ['scale', 'auto'] + expspace([-5, 1])
        },
        'poly': {
            "C": expspace([-9, 5]),
            'degree': [2, 3, 5],
            'gamma': ['scale', 'auto']
        },
        'sigmoid': {
            "C": expspace([-9, 5]),
            'gamma': ['scale', 'auto', 1, 0.1, 0.01],
            'coef0': [-10, -1, 0, 1, 10]
        }
    }
    regression_grids = {
        'linear': {
            'C':  expspace([-9, 5]),
            'epsilon': [0.01, 0.1, 0.2]
        },
        'rbf': {
            'C':  expspace([-9, 5]),
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        },
        'poly': {
            'C':  expspace([-9, 5]),
            'degree': [2, 3],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1]
        },
        'sigmoid': {
            'C':  expspace([-9, 5]),
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'epsilon': [0.01, 0.1]
        }
    }
    
    return {
        'classification': classification_grids,
        'regression': regression_grids
    }


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


def train_svm_model(input_file, 
					model_path, 
					spare_type, 
					target_column, 
					kernel, 
                    class_balancing,
					tune_hyperparameters, 
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
    
    if spare_type.isin(['BA']):
        model, scaler = None
        # Regression model (BA - Brain Age)
        if kernel=='linear':
            print("Training model with LinearSVR...")
            model, scaler = pipeline_module.train_linearsvr_model(
                dataframe=df,
                target_column=target_column,
                kernel=kernel,
                tune_hyperparameters=tune_hyperparameters,
                train_whole_set=train_whole_set
            )
        else:
            # Standard training
            print(f"Training model with default SVR with {kernel} kernel...")
            model, scaler = pipeline_module.train_svr_model(
                dataframe=df,
                target_column=target_column,
                kernel=kernel,
                tune_hyperparameters=False,
                train_whole_set=train_whole_set
            )
        
        if model != None & scaler != None & model_path != None:
            # Save model
            save_svm_model(model, scaler, {}, model_path)
            print(f"Model saved to: {model_path}")
    
    elif spare_type.isin(['AD']):
        
        if kernel.lower()=='linear':
            print("Training model with LinearSVR...")
            model, feature_encoder, label_encoder, scaler = pipeline_module.train_linearsvc_model()
        
        elif kernel.lower().isin(['poly', 'rbf', 'sigmoid']):
            model, feature_encoder, label_encoder, scaler = pipeline_module.train_svc_model(dataframe=df,
                                                                                            target_column=target_column,
                                                                                            kernel=kernel,
                                                                                            tune_hyperparameters=False,
                                                                                            class_balancing=class_balancing,
                                                                                            get_cv_scores=True,
                                                                                            train_whole_set=train_whole_set)
            pipeline_module.save_svm_model(model = model, 
                                           scaler = scaler, 
                                           training_info = {}, 
                                           filepath = model_path)
            
            print(f"Model saved to: {model_path}")
        
        else:
            print(f"Unsupported SVM kernel entry. Please select among: linear, poly, rbf, sigmoid.")
    
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
    
    # if label_encoder is not None:
    #     model_data['label_encoder'] = label_encoder
    
    joblib.dump(model_data, filepath)


def load_svm_model(filepath: str) -> Tuple[Any, StandardScaler, dict, Optional[LabelEncoder]]:
    """Load a trained model and components from a file."""
    model_data = joblib.load(filepath)
    
    model = model_data['model']
    scaler = model_data['scaler']
    training_info = model_data['training_info']
    # label_encoder = model_data.get('label_encoder', None)
    
    return model, scaler, training_info #, label_encoder
