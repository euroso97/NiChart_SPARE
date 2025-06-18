"""
NiChart_SPARE Pipelines

This package contains the pipeline modules for different SPARE score calculations.
"""

# Import pipeline modules
try:
    from . import spare_ad
    from . import spare_ba
    from . import spare_ht
except ImportError:
    # Pipeline modules might not be implemented yet
    pass

__all__ = [
    "spare_ad",
    "spare_ba", 
    "spare_ht",
] 