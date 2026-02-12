# EY Clean Water Challenge V1.0

A machine learning pipeline for predicting water quality parameters using satellite imagery, climate data, and geospatial features. This project was developed for the EY Data Challenge focused on water quality monitoring in South Africa.

## 🎯 Project Overview

This project implements an end-to-end ML solution that predicts three critical water quality parameters:
- **ALKALINITY**: Water buffering capacity
- **EC (Electrical Conductivity)**: Dissolved salt concentration
- **DRP (Dissolved Reactive Phosphorus)**: Nutrient pollution indicator

### Key Features

- 🛰️ **Multi-source Data Integration**: Combines Landsat satellite imagery, TerraClimate data, and water quality samples
- 🔬 **Advanced Feature Engineering**: Creates 100+ features including spectral indices, temporal patterns, and spatial characteristics
- 🤖 **Optimized ML Pipeline**: XGBoost models with hyperparameter tuning and spatial cross-validation
- 📊 **Comprehensive Analysis**: Statistical validation, correlation analysis, and performance monitoring
- 🚀 **Production Ready**: Snowflake integration, MLOps practices, and automated documentation

## 📁 Project Structure

```
EYCleanWaterV1.0/
├── config/
│   ├── default_config.yaml      # Configuration parameters
│   └── project.yaml             # Feature and modeling config
├── data/
│   ├── raw/                     # Raw datasets (train/test CSV, satellite data)
│   ├── processed/               # Engineered features (parquet)
│   └── external/                # Geospatial data (DEM, land cover)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── wqsa/
│   │   ├── data/                # Data loading utilities
│   │   ├── features/            # Feature engineering modules
│   │   ├── models/              # Model training and evaluation
│   │   ├── utils/               # Configuration and logging
│   │   └── visualization/       # Plotting utilities
│   └── ms_health_analytics/     # (Legacy/reference code)
├── tests/
│   ├── test_config.py
│   ├── test_features_smoke.py
│   └── test_modeling_smoke.py
├── artifacts/
│   ├── submission.csv           # Competition submission
│   ├── MODEL_CARD.md            # Model documentation
│   └── BUSINESS_PLAN.md         # Business strategy
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda package manager
- Git
- (Optional) Snowflake account for cloud deployment

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd EYCleanWaterV1.0
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n cleanwater python=3.9
conda activate cleanwater
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package**
```bash
pip install -e .
```

### Configuration

1. **Set up environment variables** (if using Snowflake):
```bash
cp .env.example .env
# Edit .env with your credentials
```

2. **Review configuration files**:
- `config/project.yaml`: Feature definitions and modeling parameters
- `config/default_config.yaml`: General project settings

## 📊 Usage

### 1. Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Explores:
- Water quality sample distributions
- Missing data patterns
- Correlations between parameters
- Temporal and spatial patterns

### 2. Feature Engineering

```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

Creates features from:
- **Landsat**: NDVI, NDBI, NDWI, cloud coverage
- **TerraClimate**: Precipitation, runoff, soil moisture
- **Temporal**: Rolling windows, seasonality
- **Spatial**: Distance metrics, clustering

### 3. Model Training

```bash
jupyter notebook notebooks/03_model_training.ipynb
```

Trains models with:
- XGBoost with optimized hyperparameters
- Spatial cross-validation (GroupKFold)
- Multi-target regression
- Feature importance analysis

### 4. Evaluation & Submission

```bash
jupyter notebook notebooks/04_evaluation.ipynb
```

Generates:
- Performance metrics (R², RMSE, MAE)
- Validation plots
- Feature importance charts
- Competition submission file

### Command Line Interface

```python
# Load configuration
from src.wqsa.utils.config import load_config
config = load_config("config/project.yaml")

# Load and process data
from src.wqsa.data import DataLoader
loader = DataLoader(config)
train_data = loader.load_training_data()

# Engineer features
from src.wqsa.features import FeatureEngineer
engineer = FeatureEngineer(config)
features = engineer.create_features(train_data)

# Train model
from src.wqsa.models import WaterQualityModel
model = WaterQualityModel(config)
model.fit(features)

# Generate predictions
predictions = model.predict(test_features)
```

## 🔧 Configuration

### Feature Configuration (`config/project.yaml`)

```yaml
features:
  landsat:
    - NDVI_MEAN_B250
    - NDBI_MEAN_B1K
    - NDWI_MEAN_B250
  terraclimate:
    - PPT_M0          # Current month precipitation
    - PPT_SUM_M3      # 3-month rolling sum
    - Q_M0            # Runoff

targets:
  - ALKALINITY
  - EC
  - DRP

modeling:
  cv_splits: 5
  random_state: 42
```

## 📈 Model Performance

Expected performance metrics:
- **R² Score**: > 0.85 on validation set
- **RMSE**: Optimized per target variable
- **Cross-Validation**: Spatial GroupKFold to prevent data leakage

Key model parameters:
- `learning_rate`: 0.035
- `n_estimators`: 900
- `max_depth`: 7
- `subsample`: 0.82
- `colsample_bytree`: 0.78

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_config.py
pytest tests/test_features_smoke.py
pytest tests/test_modeling_smoke.py

# Run with coverage
pytest --cov=src tests/
```

## 📝 Documentation

- **Model Card**: See `artifacts/MODEL_CARD.md` for detailed model documentation
- **Business Plan**: See `artifacts/BUSINESS_PLAN.md` for deployment strategy
- **API Documentation**: Auto-generated from docstrings

## 🔒 Security & Best Practices

- **Credentials**: Never commit `.env` files or credentials
- **Data Privacy**: Uses only public datasets
- **Dependencies**: Regularly update with `pip install --upgrade`
- **Code Quality**: PEP 8 compliant, type hints, comprehensive docstrings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Code Standards

- Python 3.9+
- PEP 8 style guide
- Type hints for all functions
- Google-style docstrings
- Minimum 80% test coverage

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Landsat Data**: Courtesy of the U.S. Geological Survey
- **TerraClimate**: University of Idaho
- **EY Data Challenge**: Competition organizers
- **XGBoost**: Tianqi Chen and Carlos Guestrin

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [repository-url]/issues
- Email: [your-email@example.com]

## 🔄 Version History

- **v1.0** (Current) - Initial release with core functionality
  - Multi-source data integration
  - Advanced feature engineering
  - Optimized XGBoost models
  - Spatial cross-validation
  - Automated documentation

---

**Last Updated**: February 2025  
**Status**: Active Development