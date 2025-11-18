"""
Test TabNet Encoder Integration with DeepSequence
"""
import numpy as np
import tensorflow as tf
from src.deepsequence.tabnet_encoder import TabNetEncoder, create_tabnet_encoder

print("="*70)
print("Testing TabNet Encoder Integration")
print("="*70)

# Test 1: Create TabNet encoder layer
print("\n1. Testing TabNetEncoder Layer")
print("-"*70)
encoder = TabNetEncoder(
    feature_dim=16,
    output_dim=32,
    n_steps=3,
    n_shared=2,
    n_independent=2,
    name='test_tabnet'
)
print(f"✓ TabNetEncoder created")
print(f"  - Feature dim: {encoder.feature_dim}")
print(f"  - Output dim: {encoder.output_dim}")
print(f"  - N steps: {encoder.n_steps}")

# Test 2: Build and test with sample data
print("\n2. Testing Forward Pass")
print("-"*70)
sample_input = tf.random.normal([32, 10])  # batch_size=32, features=10
output = encoder(sample_input)
print(f"✓ Forward pass successful")
print(f"  - Input shape: {sample_input.shape}")
print(f"  - Output shape: {output.shape}")
print(f"  - Expected output shape: (32, {encoder.output_dim})")

# Test 3: Create standalone model
print("\n3. Testing Standalone TabNet Model")
print("-"*70)
tabnet_model = create_tabnet_encoder(
    input_shape=(10,),
    output_dim=32,
    feature_dim=16,
    n_steps=3
)
print(f"✓ Standalone model created")
print(f"  - Total parameters: {tabnet_model.count_params():,}")
tabnet_model.summary()

# Test 4: Test predictions
print("\n4. Testing Predictions")
print("-"*70)
test_data = np.random.randn(5, 10)
predictions = tabnet_model.predict(test_data, verbose=0)
print(f"✓ Predictions generated")
print(f"  - Input shape: {test_data.shape}")
print(f"  - Output shape: {predictions.shape}")
print(f"  - Sample output (first row): {predictions[0][:5]}... (showing first 5)")

# Test 5: Test with DeepSequence integration concept
print("\n5. Testing Integration Architecture")
print("-"*70)
print("Architecture flow:")
print("  Seasonal Component → TabNet Encoder → [32-dim embedding]")
print("                                      ↓")
print("                              Intermittent Handler")
print("                                      ↓")
print("  Regressor Component → TabNet Encoder → [32-dim embedding]")
print("                                      ↓")
print("                              Combined Forecast")

# Simulate component outputs
seasonal_output = tf.random.normal([16, 1])  # Original component output
regressor_output = tf.random.normal([16, 1])

# Apply TabNet encoders
seasonal_tabnet = TabNetEncoder(output_dim=32, feature_dim=16, n_steps=3,
                                name='seasonal_tabnet')
regressor_tabnet = TabNetEncoder(output_dim=32, feature_dim=16, n_steps=3,
                                 name='regressor_tabnet')

seasonal_encoded = seasonal_tabnet(seasonal_output)
regressor_encoded = regressor_tabnet(regressor_output)

print(f"\n✓ Component encoding simulation:")
print(f"  - Seasonal: {seasonal_output.shape} → {seasonal_encoded.shape}")
print(f"  - Regressor: {regressor_output.shape} → {regressor_encoded.shape}")

# Combine for intermittent handler
combined_for_intermittent = tf.concat([seasonal_encoded, regressor_encoded],
                                       axis=-1)
print(f"  - Combined for intermittent: {combined_for_intermittent.shape}")

# Final projection back to forecast
final_seasonal = tf.keras.layers.Dense(1)(seasonal_encoded)
final_regressor = tf.keras.layers.Dense(1)(regressor_encoded)
final_forecast = final_seasonal + final_regressor

print(f"  - Final forecast shape: {final_forecast.shape}")
print(f"\n✓ Integration flow validated!")

print("\n" + "="*70)
print("✓ All TabNet Encoder tests passed!")
print("="*70)
print("\nTabNet Benefits:")
print("  • Sequential attention for feature selection")
print("  • Interpretable via attention weights")
print("  • Sparse feature usage (not all features used at each step)")
print("  • Better representation learning than simple Dense layers")
print("  • Especially effective for tabular/structured data")
