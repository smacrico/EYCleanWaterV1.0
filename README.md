# EY AI & Data Challenge - Water Quality Prediction

## 2026 EY Challenge: Predicting Water Quality Parameters

This project provides a complete machine learning pipeline for predicting water quality metrics (Alkalinity, Electrical Conductivity, and Dissolved Reactive Phosphorus) using XGBoost, geospatial features, and satellite imagery.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Conda (recommended) or pip
- 8GB+ RAM recommended
- Optional: Geospatial raster data (DEM, slope, land cover)

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Clone or navigate to the project directory
cd ey-water-quality-challenge

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ey-water-quality

# Install package in development mode
pip install -e .
```

#### Option 2: Using pip
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install package
pip install -e .
```

---

## 📁 Project Structure

```
ey-water-quality-challenge/
│
├── data/
│   ├── raw/                          # Original data files
│   │   ├── training_data.parquet
│   │   ├── validation_data.parquet
│   │   └── submission_template.csv
│   ├── processed/                    # Engineered features
│   └── external/                     # Geospatial rasters
│       ├── dem_elevation.tif
│       ├── slope_map.tif
│       └── worldcover_esa.tif
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_improved_benchmark_model.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_geospatial_features.ipynb
│   ├── 04_full_training_pipeline.ipynb
│   └── 05_submission_generator.ipynb
│
├── src/                              # Source code modules
│   ├── data_loading.py
│   ├── feature_engineering.py
│   ├── geospatial_processing.py
│   ├── model_training.py
│   ├── utils.py
│   ├── snowflake_integration.py
│   └── cli_train.py
│
├── models/                           # Saved models
├── outputs/                          # Logs and submissions
│   ├── logs/
│   └── submissions/
├── configs/                          # Configuration files
│   └── train_config.yaml
├── business_plan/                    # Business documentation
│   └── roadmap.md
│
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── README.md
```

---

## 🎯 Usage

### Method 1: Command-Line Interface (Recommended for Production)

Train all models:
```bash
python -m src.cli_train
```

Train specific targets:
```bash
python -m src.cli_train --targets "Alkalinity as CaCO3 (mg/L)" "Electrical Conductivity (µS/cm)"
```

Generate submission file:
```bash
python -m src.cli_train --generate-submission
```

Use custom configuration:
```bash
python -m src.cli_train --config configs/custom_config.yaml
```

Skip feature engineering:
```bash
python -m src.cli_train --no-feature-engineering
```

View all options:
```bash
python -m src.cli_train --help
```

### Method 2: Jupyter Notebooks (Recommended for Exploration)

Run notebooks in order:

1. **01_improved_benchmark_model.ipynb**
   - Load data
   - Basic feature engineering
   - Train baseline XGBoost models
   - Evaluate performance

2. **02_feature_engineering.ipynb**
   - Create temporal features (month, day, cyclical)
   - Calculate spectral indices (NDVI, NDWI, NBR, etc.)
   - Generate climate rolling windows
   - Build interaction features

3. **03_geospatial_features.ipynb**
   - Extract elevation from DEM
   - Extract slope values
   - Extract land cover classes
   - Create terrain complexity metrics

4. **04_full_training_pipeline.ipynb**
   - Spatial cross-validation
   - Hyperparameter tuning (Optuna)
   - Train production models
   - Feature importance analysis

5. **05_submission_generator.ipynb**
   - Load trained models
   - Process validation data
   - Generate predictions
   - Create submission CSV

### Method 3: Python API

```python
from src.data_loading import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import MultiTargetTrainer
from src.utils import load_config

# Load configuration
config = load_config('configs/train_config.yaml')

# Load data
loader = DataLoader(config['paths']['raw_data_dir'])
train_df = loader.load_training_data()

# Engineer features
fe = FeatureEngineer()
train_df = fe.create_all_features(train_df, config)

# Train models
targets = config['targets']
trainer = MultiTargetTrainer(targets, params=config['xgboost_params'])
models = trainer.train_all(X_train, y_train)

# Save models
trainer.save_all('models/')
```

---

## 🔧 Configuration

Edit `configs/train_config.yaml` to customize:

- **Data paths**: Input/output directories
- **XGBoost parameters**: Model hyperparameters (optimized defaults provided)
- **Feature engineering**: Enable/disable specific features
- **Training settings**: Test split, CV folds, early stopping
- **Logging**: Verbosity and log file location

### Optimized XGBoost Parameters

The project uses pre-tuned hyperparameters:

```yaml
xgboost_params:
  max_depth: 9
  learning_rate: 0.035
  subsample: 0.82
  colsample_bytree: 0.78
  n_estimators: 900
  min_child_weight: 3
  reg_alpha: 0.1
  reg_lambda: 1.1
  objective: "reg:squarederror"
  tree_method: "hist"
  eval_metric: "rmse"
```

---

## 📊 Features

### Implemented Feature Groups

1. **Temporal Features**
   - Month, quarter, day of year
   - Cyclical encodings (sin/cos)
   - Seasonal indicators

2. **Spectral Indices** (from Landsat)
   - NDVI (Normalized Difference Vegetation Index)
   - NDWI (Normalized Difference Water Index)
   - NBR (Normalized Burn Ratio)
   - EVI (Enhanced Vegetation Index)
   - SAVI (Soil Adjusted Vegetation Index)
   - MNDWI, BSI (Bare Soil Index)
   - Tasseled Cap (Brightness, Greenness, Wetness)

3. **Climate Features** (from TerraClimate)
   - Temperature, precipitation, soil moisture
   - Rolling windows (7, 14, 30, 60, 90 days)
   - Rolling mean and standard deviation

4. **Geospatial Features**
   - Elevation (from DEM)
   - Slope
   - Land cover classification
   - Terrain ruggedness
   - Distance from centroid

5. **Interaction Features**
   - NDVI × Temperature
   - NDWI × Precipitation
   - Elevation × Slope

---

## 🗄️ Snowflake Integration

The project supports Snowflake for data storage and retrieval.

### Setup
1. Update `configs/train_config.yaml`:
```yaml
snowflake:
  enabled: true
  account: "your_account"
  user: "your_user"
  password: "your_password"
  warehouse: "COMPUTE_WH"
  database: "EY_CHALLENGE"
  schema: "WATER_QUALITY"
```

2. Use Snowflake in CLI:
```bash
python -m src.cli_train --use-snowflake
```

3. Use in Python:
```python
from src.snowflake_integration import get_snowflake_client

client = get_snowflake_client(config)
train_df = client.load_training_data()
```

---

## 📈 Model Performance

Expected performance metrics (with full feature engineering):

| Target | R² Score | RMSE | MAE |
|--------|----------|------|-----|
| Alkalinity as CaCO3 (mg/L) | 0.85+ | ~15 | ~10 |
| Electrical Conductivity (µS/cm) | 0.88+ | ~40 | ~30 |
| Dissolved Reactive Phosphorus (mg/L) | 0.80+ | ~0.05 | ~0.03 |

---

## 🧪 Testing

Run unit tests (if implemented):
```bash
pytest tests/
```

---

## 📝 Submission

Generate submission file:

```bash
# Using CLI
python -m src.cli_train --generate-submission

# Using notebook
# Run notebooks/05_submission_generator.ipynb
```

Output: `outputs/submissions/submission_YYYYMMDD_HHMMSS.csv`

Validate submission:
```python
from src.utils import validate_submission_file

valid = validate_submission_file('outputs/submissions/submission_20260212_143000.csv')
```

---

## 🔍 Troubleshooting

### Issue: Module not found
```bash
# Ensure project is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/ey-water-quality-challenge"
```

### Issue: Geospatial features fail
- Geospatial extraction requires raster files in `data/external/`
- If unavailable, the pipeline automatically creates mock features
- Install rasterio: `conda install -c conda-forge rasterio`

### Issue: Memory errors
- Reduce `n_estimators` in config
- Use smaller dataset for testing
- Increase system RAM or use cloud compute

---

## 📚 Dependencies

Core libraries:
- **pandas** >= 1.3.0
- **numpy** >= 1.21.0
- **scikit-learn** >= 1.0.0
- **xgboost** >= 1.5.0
- **matplotlib** >= 3.4.0
- **seaborn** >= 0.11.0

Geospatial:
- **rasterio** >= 1.2.0
- **geopandas** >= 0.10.0

Optional:
- **optuna** >= 3.0.0 (hyperparameter tuning)
- **snowflake-connector-python** >= 2.7.0
- **joblib** >= 1.1.0
- **pyyaml** >= 6.0

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📄 License

This project is for the 2026 EY AI & Data Challenge.

---

## 👥 Authors

- AI/ML Team
- Geospatial Data Science Team

---

## 🙏 Acknowledgments

- EY for hosting the 2026 AI & Data Challenge
- USGS/Landsat for satellite imagery
- TerraClimate for climate data
- ESA for WorldCover land classification
- Open-source community for tools and libraries

---

## 📞 Support

For questions or issues:
1. Check existing documentation
2. Review notebook examples
3. Examine configuration files
4. Open an issue in the repository

---

**Good luck with the EY AI & Data Challenge! 🏆**
