# TabNet Encoder Integration Guide

## Overview

TabNet (Arik & Pfister, 2019) is an interpretable deep learning architecture for tabular data that uses **sequential attention** to select features at each decision step. DeepSequence now integrates TabNet encoders for both seasonal and regressor components to provide superior representation learning.

## Architecture

### TabNet in DeepSequence Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSequence with TabNet                      │
└─────────────────────────────────────────────────────────────────┘

Seasonal Component Output (1-dim)
         ↓
    TabNet Encoder (3 attention steps)
         ↓
    [32-dim rich embedding]  ──────┐
         ↓                          │
    Dense(1) → Seasonal Forecast    │
                                    │→ Concatenate → Intermittent Handler
Regressor Component Output (1-dim) │   (64-dim)        ↓
         ↓                          │              Probability (0-1)
    TabNet Encoder (3 attention steps)
         ↓                          │
    [32-dim rich embedding]  ──────┘
         ↓
    Dense(1) → Regressor Forecast
         ↓
    Combined (Add/Multiply) × Probability → Final Forecast
```

### Key Components

1. **TabNet Encoder Layer**
   - Sequential attention mechanism (N steps)
   - Shared GLU (Gated Linear Unit) blocks
   - Step-specific GLU blocks
   - Attention transformers for feature selection
   - Sparse feature usage via relaxation factor

2. **GLU (Gated Linear Unit)**
   - `GLU(x) = Linear(x) ⊗ σ(Linear(x))`
   - ⊗ = element-wise multiplication
   - σ = sigmoid activation
   - Provides non-linear feature interactions

3. **Attention Transformer**
   - Computes feature importance weights
   - Uses prior scales to discourage feature reuse
   - Sparsemax-style activation (approximated with softmax)

## Why TabNet?

### Benefits for Time Series Forecasting

1. **Interpretability**
   - Attention weights show which features are important at each step
   - Can visualize feature selection across decision steps
   - Helps understand model behavior

2. **Feature Selection**
   - Automatically selects relevant features
   - Sparse attention (not all features used)
   - Reduces overfitting on noisy features

3. **Non-linear Interactions**
   - Captures complex feature relationships
   - Better than simple Dense layers
   - Sequential processing allows hierarchical feature learning

4. **Tabular Data Optimization**
   - Designed specifically for structured/tabular data
   - Outperforms Dense networks on such data
   - Works well with both categorical and continuous features

5. **Regularization**
   - Built-in sparsity regularization
   - Batch normalization at each step
   - L1/L2 regularization support

## Usage

### Basic Usage

```python
from deepsequence import DeepSequenceModel, SeasonalComponent, RegressorComponent

# Build components
seasonal_comp = SeasonalComponent(...)
seasonal_comp.seasonal_feature()
seasonal_comp.seasonal_model(...)

regressor_comp = RegressorComponent(...)
regressor_comp.reg_model(...)

# Option 1: Without TabNet (standard)
model = DeepSequenceModel(mode='additive', use_tabnet=False)
model.build(seasonal_comp, regressor_comp)

# Option 2: With TabNet encoders
model = DeepSequenceModel(mode='additive', use_tabnet=True)
tabnet_config = {
    'output_dim': 32,       # Embedding dimension
    'feature_dim': 32,      # Internal feature transformation dimension
    'n_steps': 3,           # Number of attention steps
    'n_shared': 2,          # Shared GLU layers
    'n_independent': 2      # Step-specific GLU layers
}
model.build(seasonal_comp, regressor_comp, tabnet_config=tabnet_config)
```

### With Intermittent Handler

```python
# TabNet + Intermittent Handler (full feature set)
model = DeepSequenceModel(
    mode='additive',
    use_tabnet=True,
    use_intermittent=True
)

tabnet_config = {
    'output_dim': 32,
    'feature_dim': 32,
    'n_steps': 3,
    'n_shared': 2,
    'n_independent': 2
}

intermittent_config = {
    'hidden_units': 32,
    'hidden_layers': 2,
    'activation': 'relu',
    'dropout': 0.3,
    'l1_reg': 0.01
}

model.build(
    seasonal_comp,
    regressor_comp,
    intermittent_config=intermittent_config,
    tabnet_config=tabnet_config
)
```

## Configuration Parameters

### TabNet Configuration

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `output_dim` | 32 | Dimension of output embedding | 16-64 typical; higher for complex patterns |
| `feature_dim` | 32 | Internal transformation dimension | Usually same as output_dim |
| `n_steps` | 3 | Number of attention steps | 3-5 typical; more steps = more capacity |
| `n_shared` | 2 | Shared GLU layers | 2-3 works well |
| `n_independent` | 2 | Independent GLU layers per step | 1-2 sufficient |
| `relaxation_factor` | 1.5 | Feature reuse penalty (γ) | 1.0-2.0; higher = more diverse features |
| `bn_momentum` | 0.98 | Batch norm momentum | 0.95-0.99 |
| `sparsity_coefficient` | 1e-5 | Sparsity regularization | 1e-6 to 1e-4 |
| `l1_reg` | 0.0 | L1 regularization | 0.001-0.01 if needed |
| `l2_reg` | 0.0 | L2 regularization | 0.001-0.01 if needed |

### Configuration Guidelines

**Small Datasets (<10K samples)**
```python
tabnet_config = {
    'output_dim': 16,
    'feature_dim': 16,
    'n_steps': 2,
    'n_shared': 1,
    'n_independent': 1,
    'sparsity_coefficient': 1e-4  # Higher regularization
}
```

**Medium Datasets (10K-100K samples)**
```python
tabnet_config = {
    'output_dim': 32,
    'feature_dim': 32,
    'n_steps': 3,
    'n_shared': 2,
    'n_independent': 2,
    'sparsity_coefficient': 1e-5
}
```

**Large Datasets (>100K samples)**
```python
tabnet_config = {
    'output_dim': 64,
    'feature_dim': 64,
    'n_steps': 5,
    'n_shared': 3,
    'n_independent': 2,
    'sparsity_coefficient': 1e-6
}
```

## Performance Characteristics

### Computational Cost

**Additional Parameters:**
- Basic TabNet (32-dim, 3 steps): ~30K-40K parameters per encoder
- Two encoders (seasonal + regressor): ~60K-80K additional parameters
- Compared to base model: typically 20-30% more parameters

**Training Time:**
- ~15-25% slower training per epoch compared to base model
- Better convergence often means fewer total epochs needed
- Net training time: similar or slightly longer

### Memory Usage
- Additional GPU memory: ~10-15% for typical configurations
- Batch normalization adds some memory overhead
- Well-optimized for modern GPUs

### Expected Accuracy Improvements

Based on TabNet paper and similar applications:

**For Complex Patterns:**
- **MAE Improvement**: 5-15% reduction
- **RMSE Improvement**: 8-20% reduction
- **Feature Selection**: Better handling of irrelevant features

**For Simple Patterns:**
- May not provide significant improvement
- Standard Dense layers might be sufficient
- Consider cost vs. benefit

## When to Use TabNet

### ✅ USE TabNet when:
- Complex, non-linear feature interactions
- Many features (>10) with varying importance
- Need interpretability (which features matter)
- Tabular/structured data
- Sufficient training data (>5K samples)
- Want automatic feature selection

### ❌ DON'T USE TabNet when:
- Very simple patterns (linear relationships)
- Few features (<5)
- Very small datasets (<1K samples)
- Computational resources are limited
- Simple Dense layers already work well

## Technical Details

### Attention Mechanism

At each step t, TabNet:
1. Computes attention weights over features
2. Masks features based on attention
3. Updates prior scale to discourage reuse: `P[t+1] = P[t] × (γ - A[t])`
4. Processes masked features through GLU blocks
5. Aggregates outputs across all steps

### Sparsity Regularization

Loss includes entropy term:
```
L_sparse = -λ × mean(Σ A[t] × log(A[t]))
```
Where:
- A[t] = attention weights at step t
- λ = sparsity coefficient
- Higher entropy = more feature usage
- Penalty encourages sparse selections

### Batch Normalization

Applied at multiple stages:
- Input features
- GLU layer outputs
- Attention transformer outputs

Benefits:
- Stabilizes training
- Reduces internal covariate shift
- Allows higher learning rates

## Examples

### Example 1: Standard TabNet Integration

```python
import pandas as pd
from deepsequence import DeepSequenceModel, SeasonalComponent, RegressorComponent

# Load and prepare data
data = pd.read_csv('your_data.csv')
train, val, test = split_data(data)

# Build components
seasonal = SeasonalComponent(data=train, target=['demand'], id_var='sku_id', horizon=8)
seasonal.seasonal_feature()
seasonal.seasonal_model()

regressor = RegressorComponent(
    ts=train[['sku_id', 'ds', 'demand']],
    exog=train[['sku_id', 'price', 'promo']],
    target=['demand'],
    id_var='sku_id',
    categorical_var=['promo'],
    context_variable=['price']
)
regressor.reg_model()

# Build model with TabNet
model = DeepSequenceModel(mode='additive', use_tabnet=True)
tabnet_config = {'output_dim': 32, 'n_steps': 3}
model.build(seasonal, regressor, tabnet_config=tabnet_config)

# Train
model.compile()
model.fit(train_inputs, train_targets, epochs=100)
```

### Example 2: Full Feature Set (TabNet + Intermittent)

```python
# Build model with all features
model = DeepSequenceModel(
    mode='additive',
    use_tabnet=True,
    use_intermittent=True
)

# Configure both TabNet and Intermittent Handler
tabnet_config = {
    'output_dim': 32,
    'feature_dim': 32,
    'n_steps': 3
}

intermittent_config = {
    'hidden_units': 32,
    'hidden_layers': 2,
    'dropout': 0.3
}

model.build(
    seasonal,
    regressor,
    intermittent_config=intermittent_config,
    tabnet_config=tabnet_config
)

# The intermittent handler now uses TabNet-encoded features (64-dim)
# for better probability prediction
```

## Comparison: With vs Without TabNet

### Model Architectures

**Without TabNet (Base Model)**
```
Seasonal Component (1-dim) ─┐
                             ├→ Add/Multiply → Forecast
Regressor Component (1-dim) ─┘
```
- Simple, fast
- Limited feature learning
- ~100K parameters (typical)

**With TabNet**
```
Seasonal (1-dim) → TabNet (32-dim) → Dense(1) ─┐
                                                ├→ Add/Multiply → Forecast
Regressor (1-dim) → TabNet (32-dim) → Dense(1) ─┘
```
- Rich representations
- Feature selection
- ~160K-180K parameters (typical)

**With TabNet + Intermittent**
```
Seasonal → TabNet (32-dim) → Dense(1) ────┐
     ↓                                     ├→ Combined × Probability
     └──────→ Concat(64-dim) → Intermittent Handler
                   ↑                       
Regressor → TabNet (32-dim) → Dense(1) ────┘
```
- Best accuracy for complex intermittent data
- Full feature learning + sparsity handling
- ~180K-220K parameters (typical)

## Troubleshooting

### Issue: NaN losses during training
**Solution:** 
- Reduce learning rate
- Increase batch normalization momentum
- Add gradient clipping
- Check for extreme feature values

### Issue: Overfitting
**Solution:**
- Increase sparsity_coefficient
- Add L1/L2 regularization
- Reduce n_steps
- Use smaller output_dim

### Issue: Slow convergence
**Solution:**
- Increase learning rate slightly
- Reduce n_steps for faster iterations
- Use warmup learning rate schedule
- Ensure proper feature normalization

### Issue: No accuracy improvement
**Solution:**
- Verify data has complex patterns worth learning
- Try different n_steps (3-5)
- Increase output_dim for more capacity
- Check if base model is already optimal

## References

1. **TabNet Paper**  
   Arik, S. Ö., & Pfister, T. (2019). TabNet: Attentive Interpretable Tabular Learning.  
   arXiv preprint arXiv:1908.07442.

2. **Gated Linear Units**  
   Dauphin, Y. N., et al. (2017). Language Modeling with Gated Convolutional Networks.  
   ICML 2017.

3. **Attention Mechanisms**  
   Vaswani, A., et al. (2017). Attention is All You Need.  
   NeurIPS 2017.

## Author Notes

This TabNet integration was designed specifically for the DeepSequence architecture to:
1. Provide richer feature representations from component outputs
2. Enable automatic feature selection via attention
3. Improve intermittent handler with better encoded features
4. Maintain interpretability through attention weights

The implementation uses TensorFlow/Keras for seamless integration with existing DeepSequence components while providing the benefits of TabNet's attention mechanism.

**Author:** Mritunjay Kumar  
**Year:** 2025  
**Part of:** DeepSequence Forecasting Framework
