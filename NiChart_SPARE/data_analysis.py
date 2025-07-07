import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def report_regression_metrics(y_true, y_pred):
    """Report regression metrics: MAE, MSE, RMSE, MAPE, sMAPE, R2, Adjusted R2."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n = len(y_true)
    k = 1 if y_pred.ndim == 1 else y_pred.shape[1]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape,
        'R2': r2,
        'Adjusted R2': adj_r2
    }


def report_classification_metrics(y_true, y_pred):
    """Report classification metrics: ROC-AUC, Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, Recall, F1."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n_classes = len(np.unique(y_true))
    #metrics = {}

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary' if n_classes == 2 else 'weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary' if n_classes == 2 else 'weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary' if n_classes == 2 else 'weighted', zero_division=0)

    # Sensitivity (Recall for positive class)
    if n_classes == 2:
        sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    else:
        sensitivity = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # Specificity (Recall for negative class)
    if n_classes == 2:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except Exception:
            specificity = None
    else:
        specificity = None  # Not well-defined for multiclass

    # ROC-AUC
    try:
        if n_classes == 2:
            roc_auc = roc_auc_score(y_true, y_pred)
        else:
            roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    except Exception:
        roc_auc = None

    return {
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'ROC-AUC': roc_auc
    }

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


# Calculate the effect size of a disease for brain age model
# Optional figure generation for different disease classes
def ba_disease_effect_analysis(df, 
                               age_column = 'Age',
                               ba_columns = 'SPARE_BA',
                               disease_column='DX', 
                               export_figure=False):
    return 