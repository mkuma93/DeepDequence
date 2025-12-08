

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Concatenate, Add, Multiply, Dropout,
    LayerNormalization, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import UnitNorm
import keras
import tensorflow_recommenders as tfrs


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

@tf.function
def mish(x):
    """
    Mish activation: x * tanh(softplus(x))
    """
    return x * tf.math.tanh(tf.math.softplus(x))


@tf.function
def sparse_amplify(x):
    """
    Sparse amplify: x * 1/(abs(x)+1)
    
    Designed for sparse intermittent demand (90% zeros).
    Dampens large values, maintains small signals.
    """
    return x * (1.0 / (tf.abs(x) + 1.0))


@tf.function
def sparse_amplify_exp(x):
    """
    Sparse amplify with exponential: x * exp(1/(abs(x)+1))
    
    More aggressive sparse signal amplification:
    - x ≈ 0: amplifies by ~2.7x (exp(1) ≈ 2.718)
    - x large: no amplification (exp(0) = 1)
    
    Use for extremely sparse data requiring signal boost.
    """
    return x * tf.exp(1.0 / (tf.abs(x) + 1.0))


# ============================================================================
# CUSTOM LAYERS
# ============================================================================

@keras.saving.register_keras_serializable(package='DeepSequence')
class SqueezeLayer(tf.keras.layers.Layer):
    """Serializable layer to squeeze a specific axis."""
    
    def __init__(self, axis=1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({'axis': self.axis})
        return config


@keras.saving.register_keras_serializable(package='DeepSequence')
class ReduceSumLayer(tf.keras.layers.Layer):
    """Serializable layer to reduce sum along an axis."""
    
    def __init__(self, axis=-1, keepdims=True, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
    
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis, keepdims=self.keepdims)
    
    def get_config(self):
        config = super(ReduceSumLayer, self).get_config()
        config.update({
            'axis': self.axis,
            'keepdims': self.keepdims
        })
        return config


@keras.saving.register_keras_serializable(package='DeepSequence')
class OneMinusLayer(tf.keras.layers.Layer):
    """Serializable layer to compute 1 - x."""
    
    def call(self, inputs):
        return 1.0 - inputs
    
    def get_config(self):
        return super(OneMinusLayer, self).get_config()


@keras.saving.register_keras_serializable(package='DeepSequence')
class GatherLayer(tf.keras.layers.Layer):
    """Serializable layer to gather specific indices from input."""
    
    def __init__(self, indices, **kwargs):
        super(GatherLayer, self).__init__(**kwargs)
        self.indices = indices if isinstance(indices, list) else list(indices)
    
    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)
    
    def get_config(self):
        config = super(GatherLayer, self).get_config()
        config.update({'indices': self.indices})
        return config


@keras.saving.register_keras_serializable(package='DeepSequence')
class TemperatureSoftmax(tf.keras.layers.Layer):
    """Serializable layer for temperature-controlled softmax."""
    
    def __init__(self, temperature=1.0, **kwargs):
        super(TemperatureSoftmax, self).__init__(**kwargs)
        self.temperature = temperature
    
    def call(self, inputs):
        return tf.nn.softmax(inputs / self.temperature, axis=-1)
    
    def get_config(self):
        config = super(TemperatureSoftmax, self).get_config()
        config.update({'temperature': self.temperature})
        return config


@keras.saving.register_keras_serializable(package='DeepSequence')
class ExtractComponentWeight(tf.keras.layers.Layer):
    """Serializable layer to extract a specific component weight."""
    
    def __init__(self, component_index, **kwargs):
        super(ExtractComponentWeight, self).__init__(**kwargs)
        self.component_index = component_index
    
    def call(self, inputs):
        return tf.expand_dims(inputs[:, self.component_index], axis=-1)
    
    def get_config(self):
        config = super(ExtractComponentWeight, self).get_config()
        config.update({'component_index': self.component_index})
        return config


class MaskedEntropyAttention(tf.keras.layers.Layer):
    """
    Lightweight attention with learnable mask and entropy regularization.
    
    Key idea: Learn which features to attend to via soft masking, 
    regularized by entropy to encourage sparsity.
    
    Parameters:
    -----------
    units : int
        Output dimension after attention projection
    entropy_weight : float
        Weight for entropy regularization in loss
    dropout_rate : float
        Dropout rate for regularization
    temperature : float
        Temperature for softmax attention (default: 0.7, lower = sharper)
    
    Forward Pass:
    1. Compute attention scores: score = tanh(Wx + b)
    2. Apply temperature: score / temperature
    3. Softmax to get attention weights: α = softmax(score)
    4. Apply attention: output = α * x
    5. Project: output = Dense(output)
    6. Add entropy regularization: -Σ(α log α) to encourage sparsity
    """
    
    def __init__(self, units, entropy_weight=0.01, dropout_rate=0.1, temperature=0.7, name=None):
        super(MaskedEntropyAttention, self).__init__(name=name)
        self.units = units
        self.entropy_weight = entropy_weight
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        
    def build(self, input_shape):
        n_features = input_shape[-1]
        
        # Attention scoring layer
        self.attention_dense = Dense(
            n_features,
            activation=mish,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_attention_score'
        )
        
        # Projection after attention
        self.projection = Dense(
            self.units,
            activation=None,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_projection'
        )
        
        self.layer_norm = LayerNormalization(name=f'{self.name}_norm')
        self.dropout = Dropout(self.dropout_rate, name=f'{self.name}_dropout')
        
        super(MaskedEntropyAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: [batch, n_features]
        
        Returns:
            output: [batch, units]
            attention_weights: [batch, n_features] - for interpretability
        """
        # 1. Compute attention scores
        output = self.layer_norm(inputs)
        attention_scores = self.attention_dense(output)  # [batch, n_features]
        
        # 2. Apply temperature scaling (lower temp = more sparse)
        scaled_scores = attention_scores / self.temperature
        
        # 3. Softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_scores, axis=-1)  # [batch, n_features]
        
        # 4. Apply attention mask to input features
        masked_features = inputs * attention_weights  # [batch, n_features]
        
        # 5. Project to output dimension
        output = self.projection(masked_features)  # [batch, units]
        
        output = self.dropout(output, training=training)
        
        # 6. Compute entropy for regularization (higher entropy = less sparse)
        # We want LOW entropy (sparse attention), so we ADD entropy to loss
        epsilon = 1e-8
        entropy = -tf.reduce_sum(
            attention_weights * tf.math.log(attention_weights + epsilon),
            axis=-1
        )  # [batch]
        entropy_loss = tf.reduce_mean(entropy)
        
        # Add as model loss (will be minimized, encouraging sparsity)
        self.add_loss(self.entropy_weight * entropy_loss)
        
        # Store attention weights for interpretability
        self.last_attention_weights = attention_weights
        
        return output
    
    def get_config(self):
        config = super(MaskedEntropyAttention, self).get_config()
        config.update({
            'units': self.units,
            'entropy_weight': self.entropy_weight,
            'dropout_rate': self.dropout_rate,
            'temperature': self.temperature
        })
        return config


@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="LearnableFourierFeatures"
)
class LearnableFourierFeatures(tf.keras.layers.Layer):
    """
    Learnable Fourier features with adaptive frequencies.
    
    Instead of fixed periods (7 days, 12 months), learns optimal frequencies
    from data using gradient descent.
    
    For each frequency k:
        output = [sin(ω_k * t), cos(ω_k * t)]
    where ω_k is learnable.
    """
    
    def __init__(
        self,
        n_frequencies=5,
        initial_periods=None,
        min_period=2.0,
        max_period=365.0,
        name='learnable_fourier',
        **kwargs
    ):
        """
        Args:
            n_frequencies: Number of frequency pairs (each outputs sin + cos)
            initial_periods: List of initial periods (e.g., [7, 30, 365])
                           If None, uses log-spaced periods
            min_period: Minimum allowed period (days)
            max_period: Maximum allowed period (days)
        """
        super(LearnableFourierFeatures, self).__init__(
            name=name, **kwargs
        )
        self.n_frequencies = n_frequencies
        self.initial_periods = initial_periods
        self.min_period = min_period
        self.max_period = max_period
        
    def build(self, input_shape):
        # Initialize frequencies based on periods
        if self.initial_periods is not None:
            initial_periods = np.array(
                self.initial_periods[:self.n_frequencies],
                dtype=np.float32
            )
        else:
            # Log-spaced periods from min to max
            initial_periods = np.logspace(
                np.log10(self.min_period),
                np.log10(self.max_period),
                self.n_frequencies,
                dtype=np.float32
            )
        
        # Convert periods to frequencies: ω = 2π / period
        initial_frequencies = 2 * np.pi / initial_periods
        
        # Store log(ω) for unconstrained optimization
        # Then exponentiate to ensure positive frequencies
        self.log_frequencies = self.add_weight(
            name='log_frequencies',
            shape=(self.n_frequencies,),
            initializer=tf.keras.initializers.Constant(
                np.log(initial_frequencies)
            ),
            trainable=True,
            dtype=tf.float32
        )
        
        super(LearnableFourierFeatures, self).build(input_shape)
    
    def call(self, inputs):
        """
        Args:
            inputs: [batch, 1] time feature (typically days since epoch)
        
        Returns:
            fourier_features: [batch, 2*n_frequencies]
                [sin(ω₁t), cos(ω₁t), sin(ω₂t), cos(ω₂t), ...]
        """
        # Ensure positive frequencies
        frequencies = tf.exp(self.log_frequencies)  # [n_frequencies]
        
        # Constrain to [min_period, max_period]
        min_freq = 2 * np.pi / self.max_period
        max_freq = 2 * np.pi / self.min_period
        frequencies = tf.clip_by_value(frequencies, min_freq, max_freq)
        
        # Reshape for broadcasting: [1, n_frequencies]
        frequencies = tf.reshape(frequencies, (1, -1))
        
        # Compute ω * t: [batch, 1] * [1, n_frequencies] = [batch, n_freq]
        angles = inputs * frequencies
        
        # Compute sin and cos
        sin_features = tf.sin(angles)  # [batch, n_frequencies]
        cos_features = tf.cos(angles)  # [batch, n_frequencies]
        
        # Interleave: [sin₁, cos₁, sin₂, cos₂, ...]
        # Stack and reshape: [batch, n_freq, 2] -> [batch, 2*n_freq]
        features = tf.stack([sin_features, cos_features], axis=-1)
        features = tf.reshape(features, [tf.shape(inputs)[0], -1])
        
        return features
    
    def get_config(self):
        config = super(LearnableFourierFeatures, self).get_config()
        config.update({
            'n_frequencies': self.n_frequencies,
            'initial_periods': self.initial_periods,
            'min_period': self.min_period,
            'max_period': self.max_period
        })
        return config
    
    def get_learned_periods(self):
        """Helper to inspect learned periods after training."""
        frequencies = tf.exp(self.log_frequencies).numpy()
        periods = 2 * np.pi / frequencies
        return periods


@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="ChangepointReLU"
)
class ChangepointReLU(tf.keras.layers.Layer):
    """Vectorised ReLU transform with learnable changepoints."""

    def __init__(
        self,
        n_changepoints=10,
        time_min=0.0,
        time_max=1.0,
        name='changepoint_relu',
        **kwargs
    ):
        super(ChangepointReLU, self).__init__(name=name, **kwargs)
        self.n_changepoints = n_changepoints
        self.time_min = time_min
        self.time_max = time_max
        
    def build(self, input_shape):
        # Learn deltas between changepoints instead of absolute positions
        # This ensures they stay ordered: cp[i] = sum(deltas[0:i])
        initial_deltas = np.full(
            self.n_changepoints,
            (self.time_max - self.time_min) / self.n_changepoints,
            dtype=np.float32
        )
        
        self.changepoint_deltas = self.add_weight(
            name='changepoint_deltas',
            shape=(self.n_changepoints,),
            initializer=tf.keras.initializers.Constant(initial_deltas),
            trainable=True,
            dtype=tf.float32
        )
        super(ChangepointReLU, self).build(input_shape)

    def call(self, inputs):
        # inputs: [batch, 1] time feature
        
        # 1. Ensure deltas are positive using softplus
        positive_deltas = tf.nn.softplus(self.changepoint_deltas)
        
        # 2. Compute cumulative sum to get ordered changepoints
        changepoints = tf.cumsum(positive_deltas)
        
        # 3. Scale to [time_min, time_max] range
        time_range = self.time_max - self.time_min
        changepoints = (changepoints / changepoints[-1]) * time_range
        changepoints = changepoints + self.time_min
        
        # 4. Reshape and apply ReLU
        cp = tf.reshape(changepoints, (1, -1))  # [1, n_changepoints]
        relu_features = tf.nn.relu(inputs - cp)  # [batch, n_changepoints]
        return relu_features

    def get_config(self):
        config = super(ChangepointReLU, self).get_config()
        config.update({
            'n_changepoints': self.n_changepoints,
            'time_min': self.time_min,
            'time_max': self.time_max
        })
        return config


# ============================================================================
# Component Wrapper with SKU Embedding
# ============================================================================

@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="ComponentWithSKUWrapper"
)
class ComponentWithSKUWrapper(tf.keras.layers.Layer):
    """
    Wrapper layer that calls a component with SKU embedding.
    
    This allows components to receive both feature input and SKU embedding
    for shift-and-scale transformations in the Functional API.
    """
    
    def __init__(self, component_layer, name=None, **kwargs):
        super(ComponentWithSKUWrapper, self).__init__(name=name, **kwargs)
        self.component_layer = component_layer
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: [features, sku_embedding]
                features: Component-specific features
                sku_embedding: SKU embedding for shift-scale
        
        Returns:
            Component output
        """
        features, sku_embedding = inputs
        return self.component_layer(
            features, 
            sku_embedding=sku_embedding, 
            training=training
        )
    
    def get_config(self):
        config = super(ComponentWithSKUWrapper, self).get_config()
        config.update({
            'component_layer': keras.saving.serialize_keras_object(
                self.component_layer
            )
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        component_layer = keras.saving.deserialize_keras_object(
            config.pop('component_layer')
        )
        return cls(component_layer=component_layer, **config)


@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="TrendComponentLightweight"
)
class TrendComponentLightweight(tf.keras.layers.Layer):
    """
    Trend component matching hierarchical TabNet:
    Single time feature → ChangepointReLU → Dense → Attention → Output
    
    Learns changepoints and trend patterns via attention on hidden states.
    """
    
    def __init__(
        self,
        n_changepoints=10,
        hidden_dim=32,
        dropout_rate=0.1,
        time_min=0.0,
        time_max=1.0,
        use_sku_shift_scale=True,
        attention_temperature=0.7,
        attention_entropy_weight=0.01,
        name='trend_lightweight',
        **kwargs
    ):
        super(TrendComponentLightweight, self).__init__(
            name=name, **kwargs
        )
        self.n_changepoints = n_changepoints
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.time_min = time_min
        self.time_max = time_max
        self.use_sku_shift_scale = use_sku_shift_scale
        self.attention_temperature = attention_temperature
        self.attention_entropy_weight = attention_entropy_weight
        
    def build(self, input_shape):
        # Learnable changepoints on time feature
        self.changepoint_relu = ChangepointReLU(
            n_changepoints=self.n_changepoints,
            time_min=self.time_min,
            time_max=self.time_max,
            name=f'{self.name}_changepoints'
        )
        
        # Normalize changepoint features for stable training
        self.changepoint_norm = LayerNormalization(
            name=f'{self.name}_cp_norm'
        )
        
        # Attention mechanism: learnable weights for each changepoint
        # This learns which changepoints are important
        self.attention_layer = Dense(
            1,  # Single score per changepoint
            activation=None,
            use_bias=False,
            name=f'{self.name}_attention'
        )
        
        self.dropout_layer = Dropout(self.dropout_rate)
        
        # Dense transform on attended changepoints
        self.hidden_layer = Dense(
            self.hidden_dim,
            activation='mish',
            use_bias=False,
            name=f'{self.name}_hidden'
        )
        
        # SKU-specific shift and scale
        if self.use_sku_shift_scale:
            self.sku_beta = Dense(
                self.hidden_dim,
                activation='sigmoid',
                use_bias=False,
                name=f'{self.name}_sku_beta'
            )
            self.sku_alpha = Dense(
                self.hidden_dim,
                activation='linear',
                use_bias=False,
                name=f'{self.name}_sku_alpha'
            )
        
        # Final projection - KEEP BIAS HERE for base demand level
        self.output_layer = Dense(
            1, activation=None, use_bias=True, name=f'{self.name}_output'
        )
        
        super(TrendComponentLightweight, self).build(input_shape)
    
    def call(self, inputs, sku_embedding=None, training=None):
        """
        Args:
            inputs: [batch, 1] single time feature (date_numeric)
            sku_embedding: [batch, sku_dim] SKU embedding for shift-scale
        
        Returns:
            trend_forecast: [batch, 1]
        """
        # Apply learnable changepoints: [batch, 1] -> [batch, n_changepoints]
        cp_features = self.changepoint_relu(inputs)
        
        # Normalize changepoint features
        cp_features = self.changepoint_norm(cp_features)
        
        # Reshape to [batch, n_changepoints, 1] for attention computation
        cp_reshaped = tf.expand_dims(cp_features, axis=-1)
        
        # Compute attention score for each changepoint
        # [batch, n_changepoints, 1] -> [batch, n_changepoints, 1]
        attention_logits = self.attention_layer(cp_reshaped)
        
        # Squeeze and apply softmax with temperature
        attention_logits = tf.squeeze(attention_logits, axis=-1)
        attention_weights = tf.nn.softmax(
            attention_logits / self.attention_temperature, axis=-1
        )
        
        # Entropy regularization: encourage sparse attention
        # Lower entropy = more focused on few changepoints
        if self.attention_entropy_weight > 0:
            entropy = -tf.reduce_sum(
                attention_weights * tf.math.log(attention_weights + 1e-8),
                axis=-1
            )
            entropy_loss = self.attention_entropy_weight * tf.reduce_mean(
                entropy
            )
            self.add_loss(entropy_loss)
        
        # Weighted sum of changepoint features: [batch, n_changepoints]
        attended = cp_features * attention_weights
        attended = tf.reduce_sum(attended, axis=-1, keepdims=True)
        attended = self.dropout_layer(attended, training=training)
        
        # Dense transform to hidden representation
        trend_hidden = self.hidden_layer(attended)
        
        # Apply SKU-specific shift and scale
        if self.use_sku_shift_scale and sku_embedding is not None:
            beta = self.sku_beta(sku_embedding)
            alpha = self.sku_alpha(sku_embedding)
            trend_hidden = Multiply()([trend_hidden, beta])
            trend_hidden = Add()([trend_hidden, alpha])
        
        # Final projection
        output = self.output_layer(trend_hidden)
        
        return output


@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="SeasonalComponentLightweight"
)
class SeasonalComponentLightweight(tf.keras.layers.Layer):
    """
    Seasonal component using masked attention for Fourier features.
    
    Lighter alternative to TabNet: learns which seasonal frequencies matter.
    
    Can use either:
    - Fixed Fourier features (pre-computed)
    - Learnable Fourier features (adaptive frequencies)
    """
    
    def __init__(
        self,
        hidden_dim=32,
        dropout_rate=0.1,
        use_sku_shift_scale=True,
        use_learnable_fourier=False,
        n_learnable_frequencies=5,
        fourier_periods=None,
        activation='mish',
        name='seasonal_lightweight',
        **kwargs
    ):
        super(SeasonalComponentLightweight, self).__init__(
            name=name, **kwargs
        )
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_sku_shift_scale = use_sku_shift_scale
        self.use_learnable_fourier = use_learnable_fourier
        self.n_learnable_frequencies = n_learnable_frequencies
        self.fourier_periods = fourier_periods
        self.activation = activation
        
    def build(self, input_shape):
        # Optional: Generate Fourier features from time input
        if self.use_learnable_fourier:
            # Expects input_shape = [batch, 1] (time only)
            if self.fourier_periods is None:
                # Default: weekly, bi-weekly, monthly, quarterly, yearly
                self.fourier_periods = [7.0, 14.0, 30.0, 91.0, 365.0]
            
            # Ensure n_frequencies matches provided periods
            if len(self.fourier_periods) < self.n_learnable_frequencies:
                # If fewer periods provided, use log-spacing
                print(f"Warning: Only {len(self.fourier_periods)} periods "
                      f"provided, but n_learnable_frequencies="
                      f"{self.n_learnable_frequencies}. "
                      f"Will use log-spacing for remaining.")
                initial_periods = self.fourier_periods
            else:
                # Use exactly n_frequencies periods
                initial_periods = self.fourier_periods[
                    :self.n_learnable_frequencies
                ]
            
            self.fourier_layer = LearnableFourierFeatures(
                n_frequencies=self.n_learnable_frequencies,
                initial_periods=initial_periods,
                min_period=2.0,
                max_period=365.0,
                name=f'{self.name}_fourier'
            )
            n_features = 2 * self.n_learnable_frequencies
        else:
            # Expects pre-computed Fourier features
            n_features = input_shape[-1]
        
        # Masked attention for Fourier feature selection
        # Higher entropy (0.05): select few key frequencies
        self.attention = MaskedEntropyAttention(
            units=self.hidden_dim,
            temperature=0.5,
            entropy_weight=0.05,
            dropout_rate=self.dropout_rate,
            name=f'{self.name}_attention'
        )
        
        # Seasonal pattern learning
        self.seasonal_layer = Dense(
            self.hidden_dim // 2,
            activation=self.activation,
            use_bias=False,
            name=f'{self.name}_pattern'
        )
        
        # SKU-specific shift and scale
        if self.use_sku_shift_scale:
            self.sku_beta = Dense(
                self.hidden_dim // 2,
                activation='sigmoid',
                use_bias=False,
                name=f'{self.name}_sku_beta'
            )
            self.sku_alpha = Dense(
                self.hidden_dim // 2,
                activation='linear',
                use_bias=False,
                name=f'{self.name}_sku_alpha'
            )
        
        # Output projection
        self.output_layer = Dense(
            1, activation=None,
            use_bias=False,
            kernel_constraint=UnitNorm(axis=0),
            name=f'{self.name}_output'
        )
        
        super(SeasonalComponentLightweight, self).build(input_shape)
    
    def call(self, inputs, sku_embedding=None, training=None):
        """
        Args:
            inputs: [batch, n_fourier_features] or [batch, 1] if learnable
            sku_embedding: [batch, sku_dim] SKU embedding for shift-scale
        
        Returns:
            seasonal_forecast: [batch, 1]
        """
        # Generate Fourier features if using learnable frequencies
        if self.use_learnable_fourier:
            fourier_features = self.fourier_layer(inputs)
        else:
            fourier_features = inputs
        
        # Apply masked attention to select important frequencies
        attended_features = self.attention(
            fourier_features, training=training
        )
        
        # Learn seasonal pattern
        seasonal = self.seasonal_layer(attended_features)
        
        # Apply SKU-specific shift and scale
        if self.use_sku_shift_scale and sku_embedding is not None:
            beta = self.sku_beta(sku_embedding)
            alpha = self.sku_alpha(sku_embedding)
            seasonal = Multiply()([seasonal, beta])
            seasonal = Add()([seasonal, alpha])
        
        # Project to output
        output = self.output_layer(seasonal)
        
        return output


@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="HolidayComponentLightweight"
)
class HolidayComponentLightweight(tf.keras.layers.Layer):
    """
    Holiday component with hierarchical attention:
    1. Each holiday distance → ChangepointReLU → Individual attention
    2. Aggregate all attended outputs
    3. Apply attention on aggregated representation
    4. Final output
    """
    
    def __init__(
        self,
        n_changepoints=5,
        hidden_dim=32,
        dropout_rate=0.1,
        use_sku_shift_scale=True,
        attention_temperature=0.7,
        attention_entropy_weight=0.01,
        name='holiday_lightweight',
        **kwargs
    ):
        super(HolidayComponentLightweight, self).__init__(
            name=name, **kwargs
        )
        self.n_changepoints = n_changepoints
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_sku_shift_scale = use_sku_shift_scale
        self.attention_temperature = attention_temperature
        self.attention_entropy_weight = attention_entropy_weight
        
    def build(self, input_shape):
        n_holidays = input_shape[-1]
        self.n_holidays = n_holidays
        
        # Per-holiday changepoint layers
        self.changepoint_layers = []
        self.changepoint_norms = []  # Normalization after changepoints
        self.per_holiday_hidden = []
        self.per_holiday_attention = []
        
        for i in range(n_holidays):
            # Changepoint for each holiday distance
            cp_layer = ChangepointReLU(
                n_changepoints=self.n_changepoints,
                time_min=0.0,
                time_max=365.0,  # Holiday distances in days
                name=f'{self.name}_cp_{i}'
            )
            self.changepoint_layers.append(cp_layer)
            
            # Normalize changepoint features
            cp_norm = LayerNormalization(
                name=f'{self.name}_cp_norm_{i}'
            )
            self.changepoint_norms.append(cp_norm)
            
            # Attention on changepoint features (per holiday)
            # Dense(1) computes single attention score per changepoint
            attn_layer = Dense(
                1,  # Single score per changepoint (like Trend component)
                activation=None,
                use_bias=False,
                name=f'{self.name}_attn_{i}'
            )
            self.per_holiday_attention.append(attn_layer)
            
            # Hidden transform per holiday (after attention)
            hidden_layer = Dense(
                self.hidden_dim // n_holidays,  # Split hidden dim across holidays
                activation='mish',
                use_bias=False,
                name=f'{self.name}_hidden_{i}'
            )
            self.per_holiday_hidden.append(hidden_layer)
        
        # Aggregation attention (on concatenated attended outputs)
        self.aggregate_hidden = Dense(
            self.hidden_dim,
            activation='mish',
            use_bias=False,
            name=f'{self.name}_aggregate_hidden'
        )
        
        self.aggregate_attention = Dense(
            self.hidden_dim,
            activation=None,
            use_bias=False,
            name=f'{self.name}_aggregate_attention'
        )
        
        self.dropout_layer = Dropout(self.dropout_rate)
        
        # SKU-specific shift and scale
        if self.use_sku_shift_scale:
            self.sku_beta = Dense(
                self.hidden_dim,
                activation='sigmoid',
                use_bias=False,
                name=f'{self.name}_sku_beta'
            )
            self.sku_alpha = Dense(
                self.hidden_dim,
                activation='linear',
                use_bias=False,
                name=f'{self.name}_sku_alpha'
            )
        
        # Output projection
        self.output_layer = Dense(
            1, activation=None,
            use_bias=False,
            kernel_constraint=UnitNorm(axis=0),
            name=f'{self.name}_output'
        )
        
        super(HolidayComponentLightweight, self).build(input_shape)
    
    def call(self, inputs, sku_embedding=None, training=None):
        """
        Args:
            inputs: [batch, n_holiday_distances]
            sku_embedding: [batch, sku_dim] SKU embedding (for future use)
        
        Returns:
            holiday_forecast: [batch, 1]
        """
        attended_holidays = []
        
        # Process each holiday with its own changepoints and attention
        for i in range(self.n_holidays):
            # Extract single holiday distance: [batch, 1]
            holiday_dist = inputs[:, i:i+1]
            
            # Apply changepoints: [batch, 1] -> [batch, n_changepoints]
            cp_features = self.changepoint_layers[i](holiday_dist)
            
            # Normalize changepoint features
            cp_features = self.changepoint_norms[i](cp_features)
            
            # Reshape to [batch, n_changepoints, 1] for attention computation
            cp_reshaped = tf.expand_dims(cp_features, axis=-1)
            
            
            # Compute attention score for each changepoint
            # [batch, n_changepoints, 1] -> [batch, n_changepoints, 1]
            attn_logits = self.per_holiday_attention[i](cp_reshaped)
            
            # Apply softmax/sigmoid for relative importance
            attn_logits = tf.squeeze(attn_logits, axis=-1)  # [batch, n_changepoints]
            
            # Use sigmoid for single changepoint, softmax for multiple
            if self.n_changepoints == 1:
                attn_weights = tf.nn.sigmoid(
                    attn_logits / self.attention_temperature
                )
            else:
                attn_weights = tf.nn.softmax(
                    attn_logits / self.attention_temperature, axis=-1
                )
            
            # Entropy regularization: encourage sparse attention per holiday
            if self.attention_entropy_weight > 0 and self.n_changepoints > 1:
                entropy = -tf.reduce_sum(
                    attn_weights * tf.math.log(attn_weights + 1e-8),
                    axis=-1
                )
                entropy_loss = self.attention_entropy_weight * tf.reduce_mean(
                    entropy
                )
                self.add_loss(entropy_loss)
            
            # Weighted changepoint features (normalized attention)
            attended_cp = cp_features * attn_weights
            
            # Hidden transform on attended changepoints
            hidden = self.per_holiday_hidden[i](attended_cp)
            
            attended_holidays.append(hidden)
        
        # Concatenate all attended holiday outputs
        aggregated = tf.concat(attended_holidays, axis=-1)  # [batch, hidden_dim]
        
        # Apply aggregation-level attention with temperature
        agg_hidden = self.aggregate_hidden(aggregated)  # [batch, hidden_dim]
        agg_attn_logits = self.aggregate_attention(agg_hidden)
        agg_attn_weights = tf.nn.softmax(agg_attn_logits / self.attention_temperature, axis=-1)
        final_attended = agg_hidden * agg_attn_weights
        final_attended = self.dropout_layer(final_attended, training=training)
        
        # Apply SKU-specific shift and scale
        if self.use_sku_shift_scale and sku_embedding is not None:
            beta = self.sku_beta(sku_embedding)
            alpha = self.sku_alpha(sku_embedding)
            final_attended = Multiply()([final_attended, beta])
            final_attended = Add()([final_attended, alpha])
        
        # Project to output
        output = self.output_layer(final_attended)  # [batch, 1]
        
        return output


@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="RegressorComponentLightweight"
)
class RegressorComponentLightweight(tf.keras.layers.Layer):
    """
    Regressor component for lag features with masked attention.
    
    Learns which historical lags are most predictive.
    """
    
    def __init__(
        self,
        hidden_dim=32,
        dropout_rate=0.1,
        use_sku_shift_scale=True,
        activation='mish',
        name='regressor_lightweight',
        **kwargs
    ):
        super(RegressorComponentLightweight, self).__init__(
            name=name, **kwargs
        )
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_sku_shift_scale = use_sku_shift_scale
        self.activation = activation
        
    def build(self, input_shape):
        # Masked attention for lag selection
        # Lower entropy weight (0.01) to allow using most lags
        self.attention = MaskedEntropyAttention(
            units=self.hidden_dim,
            temperature=0.5,
            entropy_weight=0.01,
            dropout_rate=self.dropout_rate,
            name=f'{self.name}_attention'
        )
        
        # Autoregressive pattern learning
        self.ar_layer = Dense(
            self.hidden_dim // 2,
            activation=self.activation,
            use_bias=False,
            name=f'{self.name}_ar_pattern'
        )
        
        # SKU-specific shift and scale
        if self.use_sku_shift_scale:
            self.sku_beta = Dense(
                self.hidden_dim // 2,
                activation='sigmoid',
                use_bias=False,
                name=f'{self.name}_sku_beta'
            )
            self.sku_alpha = Dense(
                self.hidden_dim // 2,
                activation='linear',
                use_bias=False,
                name=f'{self.name}_sku_alpha'
            )
        
        # Output projection
        self.output_layer = Dense(
            1, activation=None,
            use_bias=False,
            kernel_constraint=UnitNorm(axis=0),
            name=f'{self.name}_output'
        )
        
        super(RegressorComponentLightweight, self).build(input_shape)
    
    def call(self, inputs, sku_embedding=None, training=None):
        """
        Args:
            inputs: [batch, n_lag_features]
            sku_embedding: [batch, sku_dim] SKU embedding (for future use)
        
        Returns:
            regressor_forecast: [batch, 1]
        """
        # Apply masked attention to select important lags
        attended_features = self.attention(inputs, training=training)
        
        # Learn AR pattern
        ar = self.ar_layer(attended_features)
        
        # Apply SKU-specific shift and scale
        if self.use_sku_shift_scale and sku_embedding is not None:
            beta = self.sku_beta(sku_embedding)
            alpha = self.sku_alpha(sku_embedding)
            ar = Multiply()([ar, beta])
            ar = Add()([ar, alpha])
        
        # Project to output
        output = self.output_layer(ar)
        
        return output


@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="CrossLayerLightweight"
)
class CrossLayerLightweight(tf.keras.layers.Layer):
    """
    DCN Cross layer using TensorFlow Recommenders.
    
    Learns multiplicative interactions between components using DCN v2.
    """
    
    def __init__(self, projection_dim=None, name='cross_layer', **kwargs):
        super(CrossLayerLightweight, self).__init__(name=name, **kwargs)
        self.projection_dim = projection_dim
        self._cross_layer = None
    
    def build(self, input_shape):
        # input_shape: list of [(batch, 1), (batch, 1), ...]
        n_components = len(input_shape)
        
        # Use TensorFlow Recommenders' Cross layer
        # It implements: x_0 * (W * x_l + b) + x_l
        self._cross_layer = tfrs.layers.dcn.Cross(
            projection_dim=self.projection_dim,
            name=f'{self.name}_dcn'
        )
        
        super(CrossLayerLightweight, self).build(input_shape)
    
    def call(self, component_outputs):
        """
        Args:
            component_outputs: List of [batch, 1] tensors from each component
        
        Returns:
            cross_output: [batch, n_components] - DCN cross features
        """
        # Stack components: [batch, n_components]
        stacked = tf.concat(component_outputs, axis=-1)
        
        # Apply DCN Cross layer
        # Output: x_0 * (W * x + b) + x  where x_0 is the input
        cross_output = self._cross_layer(stacked, stacked)
        
        return cross_output
    
    def compute_output_shape(self, input_shape):
        """DCN Cross maintains input dimension."""
        n_components = len(input_shape)
        batch_size = input_shape[0][0]
        return (batch_size, n_components)


@keras.saving.register_keras_serializable(
    package="deepsequence_hierarchical_attention",
    name="IntermittentHandlerLightweight"
)
class IntermittentHandlerLightweight(tf.keras.layers.Layer):
    """
    Lightweight intermittent demand handler.
    
    Predicts zero probability using component-level attention.
    """
    
    def __init__(self, hidden_dim=16, dropout_rate=0.1, name='intermittent_lightweight', **kwargs):
        super(IntermittentHandlerLightweight, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        # Determine input feature count
        # When loading from saved model, input_shape can be:
        # - A tuple: (None, n_features) - from CrossLayer output
        # - A list of tuples: [(None, 1), (None, 1), ...] - direct components
        if isinstance(input_shape, list):
            # Check if it's a list of shapes (direct component inputs)
            if all(isinstance(s, (tuple, list)) for s in input_shape):
                # List of component outputs, each (None, 1)
                n_features = len(input_shape)
            else:
                # Single shape passed as list
                n_features = input_shape[-1]
        else:
            # Single tensor shape from CrossLayer: (None, n_features)
            n_features = input_shape[-1]
        
        # Create layers and explicitly build them
        self.zero_prob_layer1 = Dense(
            self.hidden_dim,
            activation='relu',
            use_bias=False,
            kernel_constraint=UnitNorm(axis=0),
            name=f'{self.name}_zero_hidden'
        )
        self.zero_prob_output = Dense(
            1,
            activation='sigmoid',
            use_bias=False,
            kernel_constraint=UnitNorm(axis=0),
            name=f'{self.name}_zero_prob'
        )
        self.dropout = Dropout(self.dropout_rate)
        
        # Explicitly build the Dense layers with their input shapes
        if n_features is not None:
            self.zero_prob_layer1.build((None, n_features))
            self.zero_prob_output.build((None, self.hidden_dim))
        
        super(IntermittentHandlerLightweight, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        # Always outputs [batch, 1]
        if isinstance(input_shape, list):
            batch_size = input_shape[0][0]
        else:
            batch_size = input_shape[0]
        return (batch_size, 1)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: [batch, n_features] tensor from CrossLayer
                OR List of [batch, 1] component forecasts
        
        Returns:
            zero_prob: [batch, 1] - probability of zero demand
        """
        # Handle both list and single tensor inputs
        if isinstance(inputs, list):
            # List of components - concatenate
            features = tf.concat(inputs, axis=-1)
        else:
            # Already a tensor (from CrossLayer interactions)
            features = inputs
        
        # Predict zero probability from features (interactions or components)
        hidden = self.zero_prob_layer1(features)
        hidden = self.dropout(hidden, training=training)
        zero_prob = self.zero_prob_output(hidden)
        
        return zero_prob


def build_hierarchical_model_lightweight(
    n_temporal_features,
    n_fourier_features,
    n_holiday_features,
    n_lag_features,
    n_skus,
    n_changepoints=25,
    hidden_dim=32,
    sku_embedding_dim=8,
    dropout_rate=0.1,
    use_cross_layers=True,
    use_intermittent=True,
    combination_mode='additive',
    activation='mish'
):
    """
    Build lightweight hierarchical model with masked entropy attention.
    
    Parameters much smaller than TabNet version:
    - TabNet version: ~320K parameters
    - Lightweight version: ~30K parameters (10x reduction)
    
    Args:
        n_temporal_features: Number of time features (day, month, cyclical, etc.)
        n_fourier_features: Number of seasonal Fourier features
        n_holiday_features: Number of holiday distance features
        n_lag_features: Number of lag features
        n_skus: Number of unique SKUs for embedding
        n_changepoints: Number of trend changepoints (default: 25)
        hidden_dim: Hidden dimension for components (default: 32)
        sku_embedding_dim: SKU embedding dimension (default: 8)
        dropout_rate: Dropout rate (default: 0.1)
        use_cross_layers: Whether to use component interaction layers
        use_intermittent: Whether to use intermittent demand handling
        combination_mode: 'additive' or 'multiplicative'
    
    Returns:
        model: Keras Model
    """
    # Inputs
    temporal_input = Input(shape=(n_temporal_features,), name='temporal_features')
    fourier_input = Input(shape=(n_fourier_features,), name='fourier_features')
    holiday_input = Input(shape=(n_holiday_features,), name='holiday_features')
    lag_input = Input(shape=(n_lag_features,), name='lag_features')
    sku_input = Input(shape=(1,), name='sku_id', dtype=tf.int32)
    
    # Input-level normalization before component splitting
    # Temporal: NO normalization (changepoints need raw time values)
    # Fourier: NO normalization (sin/cos already bounded [-1, 1])
    # Holiday & Lag: NO normalization (MaskedEntropyAttention handles it internally)
    
    # SKU embedding
    sku_embedding = Embedding(
        input_dim=n_skus,
        output_dim=sku_embedding_dim,
        name='sku_embedding'
    )(sku_input)
    sku_embedding = SqueezeLayer(axis=1)(sku_embedding)  # [batch, embedding_dim]
    
    # Component outputs
    component_outputs = []
    
    # 1. Trend component (uses raw temporal input for changepoint detection)
    trend = TrendComponentLightweight(
        n_changepoints=n_changepoints,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        name='trend'
    )(temporal_input)
    component_outputs.append(trend)
    
    # 2. Seasonal component (uses raw Fourier - already bounded)
    seasonal = SeasonalComponentLightweight(
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        activation=activation,
        name='seasonal'
    )(fourier_input)
    component_outputs.append(seasonal)
    
    # 3. Holiday component (MaskedEntropyAttention normalizes internally)
    holiday = HolidayComponentLightweight(
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        name='holiday'
    )(holiday_input)
    component_outputs.append(holiday)
    
    # 4. Regressor component (lags) (MaskedEntropyAttention normalizes internally)
    regressor = RegressorComponentLightweight(
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        activation=activation,
        name='regressor'
    )(lag_input)
    component_outputs.append(regressor)
    
    # Simple additive combination - no gating
    # base_forecast = trend + seasonal + holiday + regressor
    base_forecast = Add(name='base_forecast')(component_outputs)
    
    # Intermittent handling (optional)
    if use_intermittent:
        # Use CrossLayer for intermittent handler if enabled
        if use_cross_layers:
            # CrossLayer processes component interactions for zero_prob prediction
            component_interaction = CrossLayerLightweight(name='cross_layer')(component_outputs)
            non_zero_prob = IntermittentHandlerLightweight(
                hidden_dim=16,
                dropout_rate=dropout_rate,
                name='intermittent'
            )(component_interaction)
        else:
            # Direct component outputs to intermittent handler
            non_zero_prob = IntermittentHandlerLightweight(
                hidden_dim=16,
                dropout_rate=dropout_rate,
                name='intermittent'
            )(component_outputs)
        
        # Simple Two-Stage Architecture (NO residual bypass)
        # Stage 1: Classification - Sharp decision boundary
        # Transform sigmoid output to sharper threshold (0.5 → 0/1)
        temperature = 0.1
        decision_logits = (non_zero_prob - 0.5) / temperature
        decision = Activation(
            'sigmoid',
            name='sharp_decision'
        )(decision_logits)
        
        # Stage 2: Regression - Apply decision to base forecast
        # final = base × decision
        # When decision ≈ 1 (non-zero predicted), output base forecast
        # When decision ≈ 0 (zero predicted), suppress to zero
        final_forecast = Multiply(
            name='final_forecast'
        )([base_forecast, decision])
        
        outputs = {
            'final_forecast': final_forecast,
            'non_zero_probability': non_zero_prob,
            'base_forecast': base_forecast
        }
    else:
        outputs = {'final_forecast': base_forecast}
    
    # Build model
    model = Model(
        inputs=[temporal_input, fourier_input, holiday_input, lag_input, sku_input],
        outputs=outputs,
        name='hierarchical_attention_lightweight'
    )
    
    return model


def create_model_from_features(
    X_train,
    sku_train,
    feature_indices,
    n_skus,
    hidden_dim=64,
    sku_embedding_dim=8,
    dropout_rate=0.3,
    use_cross_layers=True,
    use_intermittent=True,
    learning_rate=0.001,
    loss_weights=None
):
    """
    Wrapper function that creates and compiles a model ready for training.
    Takes 2 inputs (features + SKU) and handles the internal splitting.
    
    Args:
        X_train: Training features array [n_samples, n_features]
        sku_train: SKU IDs array [n_samples]
        feature_indices: Dict with keys 'trend', 'seasonal', 'holiday', 'regressor'
                        Each containing list of column indices
        n_skus: Number of unique SKUs
        hidden_dim: Hidden dimension for components
        sku_embedding_dim: SKU embedding dimension
        dropout_rate: Dropout rate
        use_cross_layers: Whether to use cross-layer interactions
        use_intermittent: Whether to use intermittent handling
        learning_rate: Learning rate for Adam optimizer
        loss_weights: Dict with 'base_forecast' and 'zero_probability' weights
                     If None, uses data-driven weights
    
    Returns:
        model: Compiled Keras model ready for training
        split_fn: Function to split input features for model.fit()
    """
    from tensorflow import keras
    
    # Get feature counts
    n_temporal = len(feature_indices['trend'])
    n_fourier = len(feature_indices['seasonal'])
    n_holiday = len(feature_indices['holiday'])
    n_lag = len(feature_indices['regressor'])
    
    # Build the model
    model = build_hierarchical_model_lightweight(
        n_temporal_features=n_temporal,
        n_fourier_features=n_fourier,
        n_holiday_features=n_holiday,
        n_lag_features=n_lag,
        n_skus=n_skus,
        hidden_dim=hidden_dim,
        sku_embedding_dim=sku_embedding_dim,
        dropout_rate=dropout_rate,
        use_cross_layers=use_cross_layers,
        use_intermittent=use_intermittent
    )
    
    # Define split function for features
    def split_features(X, sku):
        """Split features into component inputs"""
        X_temporal = X[:, feature_indices['trend']]
        X_fourier = X[:, feature_indices['seasonal']]
        X_holiday = X[:, feature_indices['holiday']]
        X_lag = X[:, feature_indices['regressor']]
        return [X_temporal, X_fourier, X_holiday, X_lag, sku]
    
    # Calculate data-driven loss weights if not provided
    if loss_weights is None:
        # Assume y is available for weight calculation
        # For now, use default balanced weights
        loss_weights = {
            'base_forecast': 0.5,
            'zero_probability': 0.5
        }
    
    # Define loss functions
    def base_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    from deepsequence_hierarchical_attention.losses import composite_loss
    
    # Calculate zero rate for composite loss
    # This will be applied during compilation
    zero_rate = 0.9  # Default, should be passed or calculated
    
    zero_prob_loss = composite_loss(
        zero_rate=zero_rate,
        false_negative_weight=3.0
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'base_forecast': base_mae,
            'non_zero_probability': zero_prob_loss
        },
        loss_weights=loss_weights,
        metrics={
            'final_forecast': ['mae'],
            'base_forecast': ['mae'],
            'non_zero_probability': ['accuracy', 'binary_crossentropy']
        }
    )
    
    return model, split_features


# Example usage and parameter comparison
def compare_model_sizes(
    n_temporal=10,
    n_fourier=10,
    n_holiday=15,
    n_lag=3,
    n_skus=6099,
    hidden_dim=32,
    use_cross_layers=True,
    use_intermittent=True,
    tabnet_params=322359
):
    """
    Compare parameter counts: TabNet vs Lightweight
    
    Args:
        n_temporal: Number of temporal features
        n_fourier: Number of Fourier features
        n_holiday: Number of holiday features
        n_lag: Number of lag features
        n_skus: Number of SKUs
        hidden_dim: Hidden dimension for components
        use_cross_layers: Whether to use cross-layers
        use_intermittent: Whether to use intermittent handling
        tabnet_params: Reference TabNet parameter count for comparison
    """
    print("=" * 80)
    print("PARAMETER COMPARISON: TabNet vs Lightweight Masked Attention")
    print("=" * 80)
    
    print("\nConfiguration:")
    print(f"  Temporal features: {n_temporal}")
    print(f"  Fourier features: {n_fourier}")
    print(f"  Holiday features: {n_holiday}")
    print(f"  Lag features: {n_lag}")
    print(f"  Number of SKUs: {n_skus}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Cross-layers: {use_cross_layers}")
    print(f"  Intermittent handling: {use_intermittent}")
    
    # Build lightweight model
    model_light = build_hierarchical_model_lightweight(
        n_temporal_features=n_temporal,
        n_fourier_features=n_fourier,
        n_holiday_features=n_holiday,
        n_lag_features=n_lag,
        n_skus=n_skus,
        hidden_dim=hidden_dim,
        use_cross_layers=use_cross_layers,
        use_intermittent=use_intermittent
    )
    
    total_params = model_light.count_params()
    trainable_params = sum([
        tf.size(w).numpy() for w in model_light.trainable_weights
    ])
    
    print("\n✓ Lightweight Model (Masked Entropy Attention):")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n✓ TabNet Model (reference):")
    print(f"  Total parameters: {tabnet_params:,}")
    print(f"  Trainable parameters: {tabnet_params:,}")
    
    print("\n✓ Parameter Reduction:")
    reduction = ((tabnet_params - total_params) / tabnet_params) * 100
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Lightweight params: {total_params/tabnet_params:.1%} of TabNet")
    
    return {
        'lightweight_params': total_params,
        'tabnet_params': tabnet_params,
        'reduction_percent': reduction
    }
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES:")
    print("=" * 80)
    print("TabNet:")
    print("  - Sequential attention blocks (multiple steps)")
    print("  - Feature transformer + attentive transformer")
    print("  - Ghost batch normalization")
    print("  - ~20K params per component × 4 components = ~80K")
    print("  - Heavy but powerful")
    
    print("\nLightweight Masked Attention:")
    print("  - Single attention layer per component")
    print("  - Entropy regularization for sparsity")
    print("  - Simple dense projections")
    print("  - ~2K params per component × 4 components = ~8K")
    print("  - Fast and interpretable")
    
    print("\n" + "=" * 80)
    
    return model_light


def create_lightweight_model_simple(
    n_features,
    n_skus,
    hidden_dim=32,
    sku_embedding_dim=8,
    dropout_rate=0.1,
    use_cross_layers=True,
    use_intermittent=True,
    trend_feature_indices=None,
    seasonal_feature_indices=None,
    holiday_feature_indices=None,
    regressor_feature_indices=None,
    n_changepoints=10,
    attention_entropy_weight=0.01
):
    """
    Create lightweight model with simple feature input and
    component-wise attention.
    
    This version takes all features as a single input and uses
    feature indices to split them into components internally.
    
    Args:
        n_features: Total number of input features
        n_skus: Number of unique SKUs
        hidden_dim: Hidden dimension for components
        sku_embedding_dim: SKU embedding dimension
        dropout_rate: Dropout rate
        use_cross_layers: Whether to use cross-layer interactions
        use_intermittent: Whether to handle intermittent demand
        trend_feature_indices: List of indices for trend features
        seasonal_feature_indices: List of indices for seasonal features
        holiday_feature_indices: List of indices for holiday features
        regressor_feature_indices: List of indices for regressor features
        n_changepoints: Number of trend changepoints
        attention_entropy_weight: Weight for attention entropy
        
    Returns:
        model: Keras Model with inputs [features, sku_id]
    """
    # Default feature indices if not provided
    if trend_feature_indices is None:
        trend_feature_indices = [0]
    if seasonal_feature_indices is None:
        seasonal_feature_indices = list(range(1, 7))
    if holiday_feature_indices is None:
        holiday_feature_indices = list(range(10, 25))
    if regressor_feature_indices is None:
        regressor_feature_indices = list(range(7, 10))
    
    # Inputs - split by component type directly
    trend_input = Input(
        shape=(len(trend_feature_indices),),
        name='trend_features'
    )
    seasonal_input = Input(
        shape=(len(seasonal_feature_indices),),
        name='seasonal_features'
    )
    holiday_input = Input(
        shape=(len(holiday_feature_indices),),
        name='holiday_features'
    )
    regressor_input = Input(
        shape=(len(regressor_feature_indices),),
        name='regressor_features'
    )
    sku_input = Input(shape=(1,), name='sku_id', dtype=tf.int32)
    
    # SKU embedding
    sku_embedding = Embedding(
        input_dim=n_skus,
        output_dim=sku_embedding_dim,
        name='sku_embedding'
    )(sku_input)
    sku_embedding = SqueezeLayer(axis=1)(sku_embedding)
    
    # Component outputs
    component_outputs = []
    
    # 1. Trend component with changepoint attention
    trend = ComponentWithSKUWrapper(
        TrendComponentLightweight(
            n_changepoints=n_changepoints,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attention_entropy_weight=attention_entropy_weight,
            name='trend'
        )
    )([trend_input, sku_embedding])
    component_outputs.append(trend)
    
    # 2. Seasonal component with Fourier attention
    seasonal = ComponentWithSKUWrapper(
        SeasonalComponentLightweight(
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name='seasonal'
        )
    )([seasonal_input, sku_embedding])
    component_outputs.append(seasonal)
    
    # 3. Holiday component with per-holiday attention
    holiday = ComponentWithSKUWrapper(
        HolidayComponentLightweight(
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attention_entropy_weight=attention_entropy_weight,
            name='holiday'
        )
    )([holiday_input, sku_embedding])
    component_outputs.append(holiday)
    
    # 4. Regressor component (lag features) with attention
    regressor = ComponentWithSKUWrapper(
        RegressorComponentLightweight(
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name='regressor'
        )
    )([regressor_input, sku_embedding])
    component_outputs.append(regressor)
    
    # Cross-layer interactions (optional)
    if use_cross_layers:
        combined = Concatenate(name='combine_components')(component_outputs)
        cross_layer = Dense(
            hidden_dim,
            activation='relu',
            name='cross_layer_1'
        )(combined)
        cross_layer = Dropout(dropout_rate)(cross_layer)
        cross_layer = Dense(
            hidden_dim // 2,
            activation='relu',
            name='cross_layer_2'
        )(cross_layer)
        combined_output = Dropout(dropout_rate)(cross_layer)
    else:
        combined_output = Add(name='sum_components')(component_outputs)
    
    # Output layers
    if use_intermittent:
        # Two-stage model: base_forecast and zero_probability
        base_forecast = Dense(
            1,
            activation='softplus',
            name='base_forecast'
        )(combined_output)
        zero_probability = Dense(
            1,
            activation='sigmoid',
            name='zero_probability'
        )(combined_output)
        
        # Final forecast = zero_probability * base_forecast
        final_forecast = Multiply(name='final_forecast')(
            [zero_probability, base_forecast]
        )
        
        model = Model(
            inputs=[
                trend_input,
                seasonal_input,
                holiday_input,
                regressor_input,
                sku_input
            ],
            outputs={
                'final_forecast': final_forecast,
                'base_forecast': base_forecast,
                'zero_probability': zero_probability
            },
            name='hierarchical_attention_lightweight'
        )
    else:
        # Single output
        forecast = Dense(
            1,
            activation='softplus',
            name='forecast'
        )(combined_output)
        model = Model(
            inputs=[
                trend_input,
                seasonal_input,
                holiday_input,
                regressor_input,
                sku_input
            ],
            outputs=forecast,
            name='hierarchical_attention_lightweight'
        )
    
    return model


if __name__ == "__main__":
    # Show model comparison
    model = compare_model_sizes()
    
    # Show architecture
    print("\nModel Architecture:")
    print(model.summary())
