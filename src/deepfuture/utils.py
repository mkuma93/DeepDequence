"""
Utility functions for DeepFuture Net.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def create_time_features(df: pd.DataFrame, date_col: str = 'ds') -> pd.DataFrame:
    """
    Create time-based features from datetime column.
    
    Args:
        df: DataFrame with datetime column
        date_col: Name of the datetime column
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Week of month
    df['wom'] = df[date_col].apply(lambda x: x.day // 7)
    df['year'] = df[date_col].dt.year
    df['week_no'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    
    return df


def prepare_data(df: pd.DataFrame, 
                target_col: str,
                id_col: str,
                standardize: bool = True,
                create_lags: bool = True,
                lag_periods: List[int] = [1, 4, 52]) -> pd.DataFrame:
    """
    Prepare data for DeepFuture Net training.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        id_col: Name of ID column (e.g., StockCode)
        standardize: Whether to standardize target by ID
        create_lags: Whether to create lag features
        lag_periods: List of lag periods to create
        
    Returns:
        Prepared DataFrame
    """
    df = df.copy()
    
    # Standardization
    if standardize:
        mu = df.groupby([id_col])[target_col].mean().reset_index()
        mu = mu.rename(columns={target_col: 'mean'})
        std = df.groupby([id_col])[target_col].std().reset_index()
        std = std.rename(columns={target_col: 'std'})
        
        df = df.merge(mu, on=[id_col], how='left')
        df = df.merge(std, on=[id_col], how='left')
        df[f't{target_col}'] = (df[target_col] - df['mean']) / df['std']
        target_col = f't{target_col}'
    
    # Create lag features
    if create_lags:
        for lag in lag_periods:
            df[f'lag{lag}'] = df.groupby(id_col)[target_col].shift(lag)
    
    # Fill missing values
    df.fillna(-1, inplace=True)
    
    return df


def train_val_test_split(df: pd.DataFrame,
                         date_col: str = 'ds',
                         val_weeks: int = 8,
                         test_weeks: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        val_weeks: Number of weeks for validation
        test_weeks: Number of weeks for test
        
    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    
    # Calculate split dates
    test_start = max_date - pd.Timedelta(weeks=test_weeks)
    val_start = test_start - pd.Timedelta(weeks=val_weeks)
    
    # Split data
    train = df[df[date_col] < val_start].copy()
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)].copy()
    test = df[df[date_col] >= test_start].copy()
    
    return train, val, test


def inverse_transform(predictions: np.ndarray,
                     mean: float,
                     std: float) -> np.ndarray:
    """
    Inverse transform standardized predictions back to original scale.
    
    Args:
        predictions: Standardized predictions
        mean: Mean value used for standardization
        std: Standard deviation used for standardization
        
    Returns:
        Original scale predictions
    """
    return predictions * std + mean


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def encode_categorical(df: pd.DataFrame, 
                       col: str,
                       encoder = None) -> Tuple[pd.DataFrame, object]:
    """
    Encode categorical column using ordinal encoding.
    
    Args:
        df: Input DataFrame
        col: Column to encode
        encoder: Pre-fitted encoder (if None, creates new one)
        
    Returns:
        Tuple of (DataFrame with encoded column, encoder object)
    """
    try:
        import category_encoders as ce
    except ImportError:
        raise ImportError("Please install category_encoders: pip install category-encoders")
    
    df = df.copy()
    
    if encoder is None:
        encoder = ce.OrdinalEncoder()
        df[f'{col}_encoded'] = encoder.fit_transform(df[col])
    else:
        df[f'{col}_encoded'] = encoder.transform(df[col])
    
    return df, encoder
