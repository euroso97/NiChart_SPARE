"""
NiChart_SPARE - SPARE scores calculation from Brain ROI Volumes

This package provides tools for calculating SPARE scores from brain ROI volumes
and white matter lesion volumes using machine learning models.

Supported SPARE scores:
- SPARE-CL: Classification tasks
- SPARE-RG: Regression tasks
- SPARE-BA: Brain Age
- SPARE-AD: Alzheimer's Disease
- SPARE-HT: Hypertension
- SPARE-HL: Hyperlipidemia
- SPARE-T2B: Diabetes (Type 2)
- SPARE-SM: Smoking
- SPARE-OB: Obesity
"""

__version__ = "0.1.0"
__author__ = "Kyunglok Baik"
__email__ = "software@cbica.upenn.edu"
__url__ = "https://github.com/CBICA/NiChart_SPARE"

# Import main functions for easy access
try:
    from .pipelines import spare_ad, spare_ba, spare_ht
except ImportError:
    # Pipeline modules might not be implemented yet
    pass

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__url__",
]
