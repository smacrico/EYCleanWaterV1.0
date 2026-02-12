import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Full-feature engineering module for the EY Challenge:
    - Robust date parsing
    - Automatic date-column detection
    - Temporal features
    - Cyclical encoding
    - Landsat spectral indices
    - Climate features (if present)
    """

    # Candidate date columns to auto-detect if not specified
    DATE_CANDIDATES = [
        "Sample Date", "sample_date", "sampleDate",
        "Date", "date", "datetime", "DateTime",
    ]

    def detect_date_column(self, df):
        """Return the first matching date column name, or raise."""
        for cand in self.DATE_CANDIDATES:
            if cand in df.columns:
                logger.info(f"Detected date column: {cand}")
                return cand
        raise ValueError(
            f"No suitable date column found. Columns available: {df.columns.tolist()}"
        )

    # ---------------------------------------------------------------------
    # TEMPORAL FEATURES
    # ---------------------------------------------------------------------
    def create_temporal_features(self, df, date_col=None):
        """Add robust temporal features."""

        df = df.copy()

        # Auto-detect if none provided
        if date_col is None:
            date_col = self.detect_date_column(df)

        # Ensure datetime
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception as e:
            raise ValueError(f"Failed to convert {date_col} to datetime: {e}")

        if df[date_col].isna().all():
            raise ValueError(
                f"Column '{date_col}' could not be parsed as datetime. "
                f"Example values: {df[date_col].head().tolist()}"
            )

        logger.info("Creating temporal features...")

        # Basic time features
        df["month"] = df[date_col].dt.month
        df["day_of_year"] = df[date_col].dt.dayofyear
        df["year"] = df[date_col].dt.year
        df["quarter"] = df[date_col].dt.quarter

        # Cyclical encodings
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # Custom seasonal encoding
        df["season"] = df["month"].map({12: 1, 1: 1, 2: 1,
                                        3: 2, 4: 2, 5: 2,
                                        6: 3, 7: 3, 8: 3,
                                        9: 4, 10: 4, 11: 4}).fillna(0)

        logger.info("Temporal features created: month, day_of_year, quarter, season, cyclical features.")

        return df

    # ---------------------------------------------------------------------
    # LANDSAT / SPECTRAL INDICES
    # ---------------------------------------------------------------------
    def create_landsat_indices(self, df):
        """
        Create Landsat indices if the required bands exist.
        (nir, red, green, swir16, swir22 should exist if your merge worked)
        """

        df = df.copy()

        # Check available bands
        has = lambda cols: all(c in df.columns for c in cols)

        # NDVI
        if has(["nir", "green"]):
            # Some datasets use red instead of green — detect it
            red_band = "red" if "red" in df.columns else "green"
            df["NDVI_manual"] = (df["nir"] - df[red_band]) / (df["nir"] + df[red_band])
            logger.info("NDVI created (nir & red/green).")

        # NDWI (water index)
        if has(["green", "nir"]):
            df["NDWI_manual"] = (df["green"] - df["nir"]) / (df["green"] + df["nir"])
            logger.info("NDWI created.")

        # NBR (burn ratio)
        if has(["nir", "swir22"]):
            df["NBR"] = (df["nir"] - df["swir22"]) / (df["nir"] + df["swir22"])
            logger.info("NBR created.")

        # NDMI (moisture index)
        if has(["nir", "swir16"]):
            df["NDMI_manual"] = (df["nir"] - df["swir16"]) / (df["nir"] + df["swir16"])
            logger.info("NDMI created.")

        return df

    # ---------------------------------------------------------------------
    # CLIMATE FEATURES
    # ---------------------------------------------------------------------
    def create_climate_features(self, df):
        """
        Add climate-based derived features:
        - Rolling precipitation (if ppt exists)
        - PET/PPT ratios
        - Temperature range
        """
        df = df.copy()

        # Example: PET to rainfall ratio
        if "pet" in df.columns:
            df["pet_log"] = np.log1p(df["pet"])
            logger.info("pet_log created.")

        # Add more later as needed
        return df


    # ---------------------------------------------------------------------
    # CLIMATE ROLLING FEATURES
    # ---------------------------------------------------------------------
    def create_climate_rolling_features(self, df, climate_cols, windows=[7, 30, 90]):
        """
        Create rolling-window climate features.
        
        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe.
        climate_cols : list
            Columns to compute rolling features on (e.g. ['pet', 'ppt']).
        windows : list
            Window sizes for rolling calculations (days).
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with new rolling features.
        """

        df = df.copy()

        # Ensure temporal sorting — rolling windows require ordered time
        # We already have a parsed datetime column after create_temporal_features()
        date_col = None
        for cand in self.DATE_CANDIDATES:
            if cand in df.columns:
                date_col = cand
                break

        if date_col is None:
            raise ValueError("Cannot compute rolling climate features: no date column found.")

        df = df.sort_values(date_col)

        for col in climate_cols:
            if col not in df.columns:
                logger.warning(f"[SKIP] Climate column '{col}' not found — skipping rolling features.")
                continue

            for w in windows:
                roll_col = f"{col}_roll_{w}"
                df[roll_col] = df[col].rolling(window=w, min_periods=1).mean()
                logger.info(f"Created rolling feature: {roll_col}")

        return df
    
    # ---------------------------------------------------------------------
    # SPATIAL FEATURES
    # ---------------------------------------------------------------------
    def create_spatial_features(self, df):
        """
        Create spatial features based on latitude and longitude.
        Adds:
        - lat_sin, lat_cos
        - lon_sin, lon_cos
        - distance_from_center
        - quadrant (categorical)
        
        Auto-detects lat/lon column names even if duplicated (Latitude_x etc.)
        """
        df = df.copy()

        # 1. Detect latitude & longitude columns
        lat_candidates = ["Latitude", "latitude", "lat", "Latitude_x", "Latitude_y"]
        lon_candidates = ["Longitude", "longitude", "lon", "Longitude_x", "Longitude_y"]

        lat_col = next((c for c in lat_candidates if c in df.columns), None)
        lon_col = next((c for c in lon_candidates if c in df.columns), None)

        if lat_col is None or lon_col is None:
            raise ValueError(
                f"Could not detect latitude/longitude columns. "
                f"Available columns: {df.columns.tolist()}"
            )

        logger.info(f"Using latitude column: {lat_col}")
        logger.info(f"Using longitude column: {lon_col}")

        lat = df[lat_col].astype(float)
        lon = df[lon_col].astype(float)

        # 2. Normalize angular features
        df["lat_sin"] = np.sin(np.radians(lat))
        df["lat_cos"] = np.cos(np.radians(lat))
        df["lon_sin"] = np.sin(np.radians(lon))
        df["lon_cos"] = np.cos(np.radians(lon))

        logger.info("Created trigonometric spatial features (lat/lon).")

        # 3. Distance from dataset centroid (for spatial bias awareness)
        lat_center = lat.mean()
        lon_center = lon.mean()

        df["distance_from_center"] = np.sqrt((lat - lat_center) ** 2 +
                                             (lon - lon_center) ** 2)

        logger.info("Created distance_from_center feature.")

        # 4. Quadrant (categorical spatial region)
        df["quadrant"] = np.select(
            [
                (lat >= lat_center) & (lon >= lon_center),  # NE
                (lat >= lat_center) & (lon < lon_center),   # NW
                (lat < lat_center) & (lon >= lon_center),   # SE
                (lat < lat_center) & (lon < lon_center),    # SW
            ],
            ["NE", "NW", "SE", "SW"],
            default="Unknown"
        )

        logger.info("Created spatial quadrant feature.")

        return df
    
    
    
    # ---------------------------------------------------------------------
    # MASTER PIPELINE
    # ---------------------------------------------------------------------
    def run_all(self, df, date_col=None):
        """Run temporal + Landsat + climate features."""
        logger.info("Running full feature engineering pipeline...")

        df = self.create_temporal_features(df, date_col=date_col)
        df = self.create_landsat_indices(df)
        df = self.create_climate_features(df)

        logger.info("Feature engineering pipeline complete.")
        return df