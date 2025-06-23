# Functions to help preparing the data for training and inference.
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Takes a df and turn it into a 
def encode_feature_df(
        df: pd.DataFrame
):
    columns_in_order = df.columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoder = LabelEncoder()
    encoded_data = encoder.fit_transform(df[object_cols])
    df_encoded = pd.DataFrame(encoded_data, columns=object_cols, index=df.index)

    return pd.concat([df[~df.columns.isin(object_cols)], df_encoded], axis=1)[columns_in_order], encoder


def preprocess_data(
    df: pd.DataFrame, 
    target_column: str,
    encode_categorical_features: bool = True,
    encode_categorical_target: bool = True,
    scale_features: bool = False
) -> Tuple[pd.DataFrame, pd.Series, Optional[LabelEncoder], Optional[LabelEncoder], Optional[StandardScaler]]:
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
    
    # Encode target labels if they're not numeric and encoding is requested
    target_encoder = None
    if encode_categorical_target and y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    return X, y, feature_encoder, target_encoder, scaler


def expspace(span: list) -> np.ndarray:
    return np.exp(np.linspace(span[0], span[1], num=int(span[1]) - int(span[0]) + 1)).tolist()

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
    
    return {
        'classification': classification_grids,
        'regression': regression_grids
    }