"""
NiChart_SPARE Utilities

Common functions used across different SPARE pipeline modules.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder

def validate_dataframe(df: pd.DataFrame, target_column: str) -> None:
    """Validate input dataframe and target column."""
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

def preprocess_data(
    df: pd.DataFrame, 
    target_column: str,
    encode_categorical: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Optional[LabelEncoder]]:
    """Preprocess data for training: handle missing values and encode categorical targets."""
    # Remove rows with missing target values
    df = df.dropna(subset=[target_column])
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # # Fill missing values in features with median
    # X = X.fillna(X.median())
    
    # Encode target labels if they're not numeric and encoding is requested
    label_encoder = None
    if encode_categorical and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    return X, y, label_encoder

def save_model(
    model, 
    scaler: StandardScaler, 
    training_info: dict, 
    filepath: str,
    label_encoder: Optional[LabelEncoder] = None
) -> None:
    """Save the trained model and components to a file."""
    model_data = {
        'model': model,
        'scaler': scaler,
        'training_info': training_info
    }
    
    if label_encoder is not None:
        model_data['label_encoder'] = label_encoder
    
    joblib.dump(model_data, filepath)

def load_model(filepath: str) -> Tuple[Any, StandardScaler, dict, Optional[LabelEncoder]]:
    """Load a trained model and components from a file."""
    model_data = joblib.load(filepath)
    
    model = model_data['model']
    scaler = model_data['scaler']
    training_info = model_data['training_info']
    label_encoder = model_data.get('label_encoder', None)
    
    return model, scaler, training_info, label_encoder

def predict_model(
    model, 
    scaler: StandardScaler, 
    df: pd.DataFrame,
    label_encoder: Optional[LabelEncoder] = None
) -> np.ndarray:
    """Make predictions using a trained model."""
    # Handle missing values
    df = df.fillna(df.median())
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Decode predictions if label encoder was used
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)
    
    return predictions

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Get feature importance from linear SVM model."""
    if model.kernel != 'linear':
        print("Feature importance is only meaningful for linear kernels")
        return pd.DataFrame()
    
    # Get feature importance from linear SVM
    importance = np.abs(model.coef_[0])
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

def create_training_info(
    model,
    test_metrics: dict,
    cv_score: Optional[float],
    best_params: Optional[dict],
    df: pd.DataFrame,
    X_features: pd.DataFrame,
    kernel: str,
    tune_hyperparameters: bool,
    final_model: bool = False
) -> dict:
    """Create training information dictionary."""
    training_info = {
        'n_samples': len(df),
        'n_features': X_features.shape[1],
        'kernel': kernel,
        'feature_names': list(X_features.columns),
        'tuned': tune_hyperparameters,
        'final_model': final_model
    }
    
    # Add test metrics
    training_info.update(test_metrics)
    
    # Add CV score and best parameters if available
    if cv_score is not None:
        training_info['cv_score'] = cv_score
    if best_params is not None:
        training_info['best_params'] = best_params
    
    return training_info
