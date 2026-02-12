"""
feature_engineering.py
EY AI & Data Challenge – Full Feature Engineering Pipeline

This module provides:
- Temporal feature engineering
- Cyclical encodings
- Landsat spectral indices (NDVI, NDWI, NDMI, NBR)
- Climate feature derivations (PET etc.)
- Rolling climate features
- Spatial features (lat/lon trigonometric transforms, centroid distance, quadrants)
- Automatic column detection
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Full feature engineering class for water quality prediction.
    """

    # Columns that may contain the date (auto-detected)
    DATE_CANDIDATES = [
        "Sample Date", "sample_date", "sampleDate",
        "Date", "date", "datetime", "DateTime"
    ]

    # Columns that may contain latitude / longitude
    LAT_CANDS = ["Latitude", "latitude", "lat", "Latitude_x", "Latitude_y"]
    LON_CANDS = ["Longitude", "longitude", "lon", "Longitude_x", "Longitude_y"]

    # -------------------------------------------------------------------------
    # DATE COLUMN DETECTION
    # -------------------------------------------------------------------------
    def detect_date_column(self, df):
        for c in self.DATE_CANDIDATES:
            if c in df.columns:
                logger.info(f"Detected date column: {c}")
                return c
        raise ValueError(
            f"No valid date column found. Available columns: {df.columns.tolist()}"
        )

    # -------------------------------------------------------------------------
    # TEMPORAL FEATURES
    # -------------------------------------------------------------------------
    def create_temporal_features(self, df, date_col=None):
        df = df.copy()

        # Auto-detect date column if not provided
        if date_col is None:
            date_col = self.detect_date_column(df)

        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        if df[date_col].isna().all():
            raise ValueError(
                f"Column '{date_col}' could not be parsed as datetime. "
                f"Example values: {df[date_col].head().tolist()}"
            )

        logger.info("Creating temporal features...")

        df["month"] = df[date_col].dt.month
        df["day_of_year"] = df[date_col].dt.dayofyear
        df["year"] = df[date_col].dt.year
        df["quarter"] = df[date_col].dt.quarter

        # Cyclical encodings
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # Simple 1–4 seasonal encoding
        df["season"] = df["month"].map({
            12: 1, 1: 1, 2: 1,     # Winter
            3: 2, 4: 2, 5: 2,     # Spring
            6: 3, 7: 3, 8: 3,     # Summer
            9: 4, 10: 4, 11: 4    # Autumn
        }).fillna(0)

        logger.info("Temporal features created successfully.")

        return df

    # -------------------------------------------------------------------------
    # LANDSAT SPECTRAL INDICES
    # -------------------------------------------------------------------------
    def create_landsat_indices(self, df):
        df = df.copy()

        def has(cols):
            return all(c in df.columns for c in cols)

        # NDVI: (NIR - RED) / (NIR + RED)
        if "nir" in df.columns:
            red_band = "red" if "red" in df.columns else ("green" if "green" in df.columns else None)
            if red_band:
                df["NDVI"] = (df["nir"] - df[red_band]) / (df["nir"] + df[red_band])
                logger.info("Created NDVI.")

        # NDWI: (GREEN - NIR) / (GREEN + NIR)
        if has(["green", "nir"]):
            df["NDWI"] = (df["green"] - df["nir"]) / (df["green"] + df["nir"])
            logger.info("Created NDWI.")

        # NBR: (NIR - SWIR2) / (NIR + SWIR2)
        if has(["nir", "swir22"]):
            df["NBR"] = (df["nir"] - df["swir22"]) / (df["nir"] + df["swir22"])
            logger.info("Created NBR.")

        # NDMI: (NIR - SWIR1) / (NIR + SWIR1)
        if has(["nir", "swir16"]):
            df["NDMI"] = (df["nir"] - df["swir16"]) / (df["nir"] + df["swir16"])
            logger.info("Created NDMI.")

        logger.info("Landsat indices completed.")
        return df

    # -------------------------------------------------------------------------
    # CLIMATE FEATURES (STATIC)
    # -------------------------------------------------------------------------
    def create_climate_features(self, df):
        df = df.copy()

        # PET log transform (useful for skewed climate distributions)
        if "pet" in df.columns:
            df["pet_log"] = np.log1p(df["pet"])
            logger.info("Created pet_log.")

        return df

    # -------------------------------------------------------------------------
    # CLIMATE ROLLING FEATURES
    # -------------------------------------------------------------------------
    def create_climate_rolling_features(self, df, climate_cols, windows=[7, 30, 90]):
        df = df.copy()

        # Identify date column so we know how to sort
        date_col = None
        for c in self.DATE_CANDIDATES:
            if c in df.columns:
                date_col = c
                break

        if date_col is None:
            raise ValueError("Cannot compute rolling climate features: no date column found.")

        df = df.sort_values(date_col)

        for col in climate_cols:
            if col not in df.columns:
                logger.warning(f"[SKIP] '{col}' missing — skipping rolling.")
                continue

            for w in windows:
                new_col = f"{col}_roll_{w}"
                df[new_col] = df[col].rolling(w, min_periods=1).mean()
                logger.info(f"Created rolling feature: {new_col}")

        return df

    # -------------------------------------------------------------------------
    # SPATIAL FEATURES
    # -------------------------------------------------------------------------
    def create_spatial_features(self, df):
        df = df.copy()

        lat_col = next((c for c in self.LAT_CANDS if c in df.columns), None)
        lon_col = next((c for c in self.LON_CANDS if c in df.columns), None)

        if lat_col is None or lon_col is None:
            raise ValueError(
                f"Latitude/longitude columns not found. "
                f"Available: {df.columns.tolist()}"
            )

        lat = df[lat_col].astype(float)
        lon = df[lon_col].astype(float)

        # Trigonometric encodings
        df["lat_sin"] = np.sin(np.radians(lat))
        df["lat_cos"] = np.cos(np.radians(lat))
        df["lon_sin"] = np.sin(np.radians(lon))
        df["lon_cos"] = np.cos(np.radians(lon))

        logger.info("Created trigonometric spatial encodings.")

        # Distance from global centroid of dataset
        lat_c = lat.mean()
        lon_c = lon.mean()
        df["distance_from_center"] = np.sqrt((lat - lat_c) ** 2 + (lon - lon_c) ** 2)

        # Quadrant classification
        df["quadrant"] = np.select(
            [
                (lat >= lat_c) & (lon >= lon_c),
                (lat >= lat_c) & (lon < lon_c),
                (lat < lat_c) & (lon >= lon_c),
                (lat < lat_c) & (lon < lon_c),
            ],
            ["NE", "NW", "SE", "SW"],
            default="UNK",
        )

        logger.info("Created spatial quadrant + distance_from_center.")

        return df
    # ---------------------------------------------------------------------
    # INTERACTION FEATURES
    # ---------------------------------------------------------------------
    def create_interaction_features(self, df):
        """
        Create meaningful interaction features between:
        - Climate × Landsat indices
        - Climate × temporal
        - Landsat × spatial
        - Spatial complexity metrics

        Safe for tree models (XGBoost, LightGBM).
        """

        df = df.copy()

        logger.info("Creating interaction features...")

        # -------------------------------------------
        # 1. Climate × Temporal Interactions
        # -------------------------------------------
        if "pet" in df.columns:
            if "month" in df.columns:
                df["pet_x_month"] = df["pet"] * df["month"]
            if "day_of_year" in df.columns:
                df["pet_x_dayofyear"] = df["pet"] * df["day_of_year"]

        # -------------------------------------------
        # 2. Landsat × Climate Interactions
        # -------------------------------------------
        landsat_cols = ["NDVI", "NDWI", "NDMI", "NBR"]
        for col in landsat_cols:
            if col in df.columns and "pet" in df.columns:
                df[f"{col}_x_pet"] = df[col] * df["pet"]

        # -------------------------------------------
        # 3. Landsat × Spatial Interactions
        # -------------------------------------------
        spatial_cols = ["lat_sin", "lat_cos", "lon_sin", "lon_cos"]

        for s in spatial_cols:
            if s in df.columns and "NDVI" in df.columns:
                df[f"NDVI_x_{s}"] = df["NDVI"] * df[s]

        # -------------------------------------------
        # 4. Terrain Complexity (based on slope, if available)
        # -------------------------------------------
        if "slope" in df.columns:
            df["terrain_complexity"] = np.sqrt(
                df["slope"].astype(float)**2 +
                (df["NDMI"] if "NDMI" in df.columns else 0)**2
            )

        # -------------------------------------------
        # 5. Distance weighting effects
        # -------------------------------------------
        if "distance_from_center" in df.columns and "NDVI" in df.columns:
            df["NDVI_weighted_distance"] = df["NDVI"] * df["distance_from_center"]

        logger.info("Interaction features created successfully.")
        return df
    
    
    
    # -------------------------------------------------------------------------
    # MASTER PIPELINE
    # -------------------------------------------------------------------------
    def run_all(self, df, date_col=None):
        logger.info("Running full feature engineering pipeline...")

        df = self.create_temporal_features(df, date_col=date_col)
        df = self.create_landsat_indices(df)
        df = self.create_climate_features(df)
        df = self.create_climate_rolling_features(df, climate_cols=["pet"])
        df = self.create_spatial_features(df)

        logger.info("Feature engineering pipeline completed successfully.")
        return df