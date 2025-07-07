"""
NiChart_SPARE Utilities
Common functions used across different SPARE pipeline modules.
"""
# import sys
import pandas as pd
import numpy as np
# import joblib
from typing import Tuple, Optional, Dict, Any, Union
# from sklearn.preprocessing import StandardScaler, LabelEncoder


# ############# MISC ################

def expspace(span: list) -> np.ndarray:
    return np.exp(np.linspace(span[0], span[1], num=int(span[1]) - int(span[0]) + 1)).tolist()

def is_regression_model(spare_type):
    """Check if the SPARE type uses regression (continuous target)"""
    regression_types = ['BA']  # Brain Age is continuous
    return spare_type.upper() in regression_types


def get_svm_hyperparameter_grids():
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