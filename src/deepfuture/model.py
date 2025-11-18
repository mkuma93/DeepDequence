"""
Main DeepFuture Net model combining seasonal and regression components.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import List, Tuple
import pandas as pd
import numpy as np

from .seasonal_component import SeasonalComponent
from .regressor_component import RegressorComponent
from .config import DEFAULT_LEARNING_RATE, TRAINING_PARAMS


class DeepFutureModel:
    """
    DeepFuture Net: Complete forecasting model combining seasonal and 
    regression components.
    """
    
    def __init__(self, 
                 mode: str = 'additive'):
        """
        Initialize DeepFuture model.
        
        Args:
            mode: Combination mode ('additive' or 'multiplicative')
        """
        self.mode = mode
        self.seasonal_model = None
        self.regressor_model = None
        self.full_model = None
        
    def build(self,
              seasonal_component: SeasonalComponent,
              regressor_component: RegressorComponent):
        """
        Build the complete DeepFuture model.
        
        Args:
            seasonal_component: Initialized seasonal component
            regressor_component: Initialized regressor component
        """
        self.seasonal_model = seasonal_component.s_model
        self.regressor_model = regressor_component.combined_reg_model
        
        # Combine seasonal and regression outputs using Keras layers
        if self.mode == 'additive':
            combined_output = layers.Add(name='additive_forecast')(
                [self.seasonal_model.output, self.regressor_model.output]
            )
        else:  # multiplicative
            combined_output = layers.Multiply(name='multiplicative_forecast')(
                [self.seasonal_model.output, self.regressor_model.output]
            )
        
        # Create combined model
        self.full_model = Model(
            inputs=[self.seasonal_model.input, self.regressor_model.input],
            outputs=combined_output,
            name='deepfuture_net'
        )
        
        return self.full_model
    
    def compile(self,
                loss='mape',
                learning_rate: float = DEFAULT_LEARNING_RATE):
        """
        Compile the model.
        
        Args:
            loss: Loss function ('mape', 'mae', 'mse', or custom)
            learning_rate: Learning rate for optimizer
        """
        if loss == 'mape':
            loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
        elif loss == 'mae':
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif loss == 'mse':
            loss_fn = tf.keras.losses.MeanSquaredError()
        else:
            loss_fn = loss
        
        self.full_model.compile(
            loss=loss_fn,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )
    
    def fit(self,
            train_input: List,
            train_target: np.ndarray,
            val_input: List = None,
            val_target: np.ndarray = None,
            epochs: int = 500,
            batch_size: int = 512,
            checkpoint_path: str = None,
            patience: int = 10,
            verbose: int = 1):
        """
        Train the model.
        
        Args:
            train_input: List of training inputs [seasonal_inputs, regressor_inputs]
            train_target: Training target values
            val_input: Validation inputs (optional)
            val_target: Validation target values (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            checkpoint_path: Path to save best model
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss' if val_input else 'loss',
            mode=TRAINING_PARAMS['mode'],
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stop)
        
        # Model checkpoint
        if checkpoint_path:
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if val_input else 'loss',
                save_best_only=TRAINING_PARAMS['save_best_only'],
                mode=TRAINING_PARAMS['mode']
            )
            callbacks.append(checkpoint)
        
        # Training
        validation_data = None
        if val_input is not None and val_target is not None:
            validation_data = (val_input, val_target)
        
        history = self.full_model.fit(
            train_input,
            train_target,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, inputs: List) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            inputs: List of inputs [seasonal_inputs, regressor_inputs]
            
        Returns:
            Predictions array
        """
        return self.full_model.predict(inputs)
    
    def save(self, path: str):
        """
        Save the model.
        
        Args:
            path: Path to save model
        """
        self.full_model.save(path)
    
    @staticmethod
    def load(path: str, custom_objects: dict = None):
        """
        Load a saved model.
        
        Args:
            path: Path to saved model
            custom_objects: Dictionary of custom objects
            
        Returns:
            Loaded DeepFuture model
        """
        model = DeepFutureModel()
        model.full_model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        return model
    
    def summary(self):
        """Print model summary."""
        if self.full_model:
            self.full_model.summary()
        else:
            print("Model not built yet. Call build() first.")
