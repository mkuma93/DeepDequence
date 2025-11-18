"""
DeepFuture Net: A Prophet-inspired deep learning architecture for time series forecasting.

This package provides modules for building and training the DeepFuture Net architecture,
which decomposes forecasting into seasonal and regression components.

Author: Mritunjay Kumar
Year: 2021
"""

__version__ = "0.1.0"
__author__ = "Mritunjay Kumar"

from .seasonal_component import SeasonalComponent
from .regressor_component import RegressorComponent
from .model import DeepFutureModel
from .utils import create_time_features, prepare_data

__all__ = [
    'SeasonalComponent',
    'RegressorComponent', 
    'DeepFutureModel',
    'create_time_features',
    'prepare_data'
]
