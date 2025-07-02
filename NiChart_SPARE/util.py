"""
NiChart_SPARE Utilities

Common functions used across different SPARE pipeline modules.
"""
# import sys
import pandas as pd
import numpy as np
# import joblib
# from typing import Tuple, Optional, Dict, Any, Union
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# ############# MISC ################

def expspace(span: list) -> np.ndarray:
    return np.exp(np.linspace(span[0], span[1], num=int(span[1]) - int(span[0]) + 1)).tolist()

def is_regression_model(spare_type):
    """Check if the SPARE type uses regression (continuous target)"""
    regression_types = ['BA']  # Brain Age is continuous
    return spare_type.upper() in regression_types