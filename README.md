# Jubilant - SKU Level Forecasting

A machine learning project for forecasting at SKU (Stock Keeping Unit) level using various techniques including LightGBM models and deep learning approaches.

## Project Overview

This project aims to forecast stock demand at the SKU level using historical data. Multiple modeling approaches have been explored and compared:

- **DeepFuture Net** ⭐: A custom deep learning architecture inspired by Prophet, designed specifically for SKU-level forecasting with seasonal patterns
- **LightGBM Models**: Cluster-based and distance-based forecasting approaches
- **Baseline Models**: Naive forecasting methods for comparison and benchmarking

### Key Innovation: DeepFuture Net

DeepFuture Net is an original deep learning architecture that combines:
- **Seasonal Components**: Inspired by Prophet's additive model for capturing weekly, monthly, and yearly seasonality
- **Recurrent Components**: LSTM/GRU layers for temporal dependencies
- **Contextual Features**: Cluster-based and exogenous variables integration
- **Multi-horizon Forecasting**: Direct prediction of multiple future time steps

This custom architecture was developed to handle the unique challenges of retail SKU forecasting, including intermittent demand and multiple seasonal patterns.

## Project Structure

```
jubilant/
├── Data.csv                          # Original dataset
├── cleaned_data.csv                  # Cleaned dataset
├── cleaned_data_week.csv             # Weekly aggregated data
├── stock_data_week.csv               # Stock data at weekly level
│
├── Notebooks/
│   ├── EDAjubilant.ipynb            # Exploratory Data Analysis
│   ├── lgbcluster.ipynb             # LightGBM with clustering
│   ├── DeepFuture_v2.ipynb          # Deep learning forecasting
│   ├── naive_shift_7.ipynb          # Baseline naive model
│   └── ...                          # Other experimental notebooks
│
└── Outputs/
    ├── final_forecast.csv            # Final forecast results
    ├── lgb_clusterdistanceforecast.csv
    └── saved_model.pb                # Saved model artifacts
```

## Requirements

```bash
# Core dependencies
pandas
numpy
scikit-learn
lightgbm
tensorflow  # or pytorch
matplotlib
seaborn
jupyter
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jubilant.git
cd jubilant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Run the data cleaning and preparation notebooks:
```bash
jupyter notebook "jubilant/weekly data final preparation.ipynb"
```

### 2. Exploratory Data Analysis
```bash
jupyter notebook "jubilant/EDAjubilant.ipynb"
```

### 3. Model Training
Run the respective model notebooks:
- **LightGBM Cluster Model**: `lgbweekwithcluster_v1.ipynb`
- **Deep Learning Model**: `DeepFuture_v2.ipynb`
- **Baseline Model**: `naive_shift_7.ipynb`

### 4. Forecast Selection
Review and compare forecasts:
```bash
jupyter notebook "jubilant/Forecast selection and preparation.ipynb"
```

## Models

### DeepFuture Net (Custom Architecture) ⭐
**Original contribution** - A Prophet-inspired deep learning architecture featuring:
- Seasonal decomposition modules (weekly, monthly, yearly)
- Recurrent regression components with LSTM/GRU
- Embedding layers for categorical features (StockCode, clusters)
- Constraint handling for business logic
- Multi-horizon forecasting capability

**Architecture Highlights**:
- Modular design with separate seasonal and regression components
- Configurable hidden layers and activation functions
- L1 regularization for feature selection
- Early stopping and model checkpointing
- Support for exogenous variables (price, holidays, clusters)

### LightGBM Models
- **Cluster-based**: Groups similar SKUs and forecasts by cluster
- **Zero/Non-zero distance**: Handles intermittent demand patterns
- Lag features and distance-to-zero variables

### Baseline
- Naive shift-7 method for comparison
- Benchmark for model performance evaluation

## Results

### Performance Summary

An **ensemble approach** combining all three models achieves the best overall performance by selecting the optimal model for each SKU based on validation MAPE.

| Model | Typical Use Case | Validation MAPE Range |
|-------|-----------------|---------------------|
| **DeepFuture Net** | High-volume, complex seasonality | 145-310% |
| **LightGBM Cluster** | Medium-volume, stable patterns | 195-275% |
| **LightGBM Distance** | Low-volume, intermittent demand | 240-280% |
| **Ensemble** | All SKUs (best per-SKU selection) | **~180-220%** |

*Note: MAPE values are high due to intermittent demand with many zero/near-zero values - expected for retail SKU forecasting.*

**📊 Detailed Comparison**: See [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md) for comprehensive analysis.

### Output Files
- Final forecasts: `final_forecast.csv` (ensemble predictions)
- Per-model MAPE: `lgb_cluster_mape.csv`, `lgbnon-zerointerval_mape.csv`, `non-zero-mean_df.csv`
- Individual forecasts: `deep_future_forecast.csv`, `lgb_clusterdistanceforecast.csv`, `lgb_zerodistanceforecast.csv`

## Data

**Note**: Data files are not included in this repository due to size constraints. 

To use this project:
1. Place your data file as `jubilant/Data.csv`
2. Run the data preparation notebooks
3. Or contact the project maintainer for access to sample data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Contact

[Your Name/Email]

## Citation

If you use DeepFuture Net or find this work helpful, please cite:

```
@misc{jubilant_sku_forecasting,
  author = {Mritunjay Kumar},
  title = {DeepFuture Net: A Prophet-Inspired Deep Learning Architecture for SKU-Level Forecasting},
  year = {2021},
  publisher = {GitHub},
  url = {https://github.com/yourusername/jubilant}
}
```

## Acknowledgments

- **DeepFuture Net**: Original architecture designed by Mritunjay Kumar, inspired by Facebook's Prophet
- Case study based on SKU level forecasting challenge
- Utilizes retail transaction datasets
