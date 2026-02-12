"""
Geospatial Processing Module for EY Water Quality Challenge
Extract elevation, slope, and land cover features from raster data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import logging

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_geom
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("rasterio not available. Geospatial features will be limited.")

logger = logging.getLogger(__name__)


class GeospatialProcessor:
    """Extract geospatial features from raster data"""
    
    def __init__(self, 
                 dem_path: Optional[str] = None,
                 slope_path: Optional[str] = None,
                 landcover_path: Optional[str] = None):
        """
        Initialize geospatial processor
        
        Args:
            dem_path: Path to DEM elevation raster
            slope_path: Path to slope raster
            landcover_path: Path to land cover raster
        """
        self.dem_path = Path(dem_path) if dem_path else None
        self.slope_path = Path(slope_path) if slope_path else None
        self.landcover_path = Path(landcover_path) if landcover_path else None
        
        if not RASTERIO_AVAILABLE:
            logger.warning("Rasterio not installed. Install with: pip install rasterio")
    
    def sample_raster_at_points(self, 
                                raster_path: Path,
                                lat_lon_points: List[Tuple[float, float]],
                                nodata_value: float = -9999) -> np.ndarray:
        """
        Sample raster values at given lat/lon coordinates
        
        Args:
            raster_path: Path to raster file
            lat_lon_points: List of (latitude, longitude) tuples
            nodata_value: Value to use for no-data pixels
            
        Returns:
            Array of sampled values
        """
        if not RASTERIO_AVAILABLE:
            logger.error("Rasterio required for raster sampling")
            return np.full(len(lat_lon_points), np.nan)
        
        if not raster_path.exists():
            logger.warning(f"Raster file not found: {raster_path}")
            return np.full(len(lat_lon_points), np.nan)
        
        try:
            with rasterio.open(raster_path) as src:
                # Convert lat/lon to row/col indices
                values = []
                for lat, lon in lat_lon_points:
                    # Note: rasterio uses (lon, lat) order for coordinates
                    row, col = src.index(lon, lat)
                    
                    # Check bounds
                    if 0 <= row < src.height and 0 <= col < src.width:
                        value = src.read(1)[row, col]
                        # Handle nodata values
                        if value == nodata_value or (src.nodata and value == src.nodata):
                            values.append(np.nan)
                        else:
                            values.append(value)
                    else:
                        values.append(np.nan)
                
                return np.array(values)
                
        except Exception as e:
            logger.error(f"Error sampling raster {raster_path}: {e}")
            return np.full(len(lat_lon_points), np.nan)
    
    def extract_elevation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract elevation from DEM at sample points
        
        Args:
            df: DataFrame with Latitude and Longitude columns
            
        Returns:
            DataFrame with elevation column added
        """
        if self.dem_path is None:
            logger.warning("DEM path not specified")
            df['elevation'] = np.nan
            return df
        
        logger.info(f"Extracting elevation from {self.dem_path}")
        
        lat_lon_points = list(zip(df['Latitude'], df['Longitude']))
        elevation_values = self.sample_raster_at_points(self.dem_path, lat_lon_points)
        
        df = df.copy()
        df['elevation'] = elevation_values
        
        # Fill missing values with median
        median_elevation = np.nanmedian(elevation_values)
        df['elevation'].fillna(median_elevation, inplace=True)
        
        logger.info(f"Elevation range: {df['elevation'].min():.2f} to {df['elevation'].max():.2f}")
        
        return df
    
    def extract_slope(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract slope from slope raster at sample points
        
        Args:
            df: DataFrame with Latitude and Longitude columns
            
        Returns:
            DataFrame with slope column added
        """
        if self.slope_path is None:
            logger.warning("Slope path not specified")
            df['slope'] = np.nan
            return df
        
        logger.info(f"Extracting slope from {self.slope_path}")
        
        lat_lon_points = list(zip(df['Latitude'], df['Longitude']))
        slope_values = self.sample_raster_at_points(self.slope_path, lat_lon_points)
        
        df = df.copy()
        df['slope'] = slope_values
        
        # Fill missing values with median
        median_slope = np.nanmedian(slope_values)
        df['slope'].fillna(median_slope, inplace=True)
        
        logger.info(f"Slope range: {df['slope'].min():.2f} to {df['slope'].max():.2f}")
        
        return df
    
    def extract_landcover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract land cover class from ESA WorldCover at sample points
        
        Args:
            df: DataFrame with Latitude and Longitude columns
            
        Returns:
            DataFrame with landcover column and one-hot encoded features
        """
        if self.landcover_path is None:
            logger.warning("Land cover path not specified")
            df['landcover'] = 0
            return df
        
        logger.info(f"Extracting land cover from {self.landcover_path}")
        
        lat_lon_points = list(zip(df['Latitude'], df['Longitude']))
        landcover_values = self.sample_raster_at_points(self.landcover_path, lat_lon_points)
        
        df = df.copy()
        df['landcover'] = landcover_values
        
        # Fill missing values with mode
        mode_landcover = df['landcover'].mode()[0] if len(df['landcover'].mode()) > 0 else 0
        df['landcover'].fillna(mode_landcover, inplace=True)
        df['landcover'] = df['landcover'].astype(int)
        
        # ESA WorldCover classes (simplified)
        # 10: Tree cover, 20: Shrubland, 30: Grassland, 40: Cropland,
        # 50: Built-up, 60: Bare/sparse vegetation, 70: Snow/ice,
        # 80: Water bodies, 90: Wetlands, 95: Mangroves, 100: Moss/lichen
        
        # One-hot encode major land cover types
        landcover_mapping = {
            10: 'forest',
            20: 'shrubland',
            30: 'grassland',
            40: 'cropland',
            50: 'urban',
            60: 'barren',
            70: 'snow',
            80: 'water',
            90: 'wetland',
            95: 'mangrove',
            100: 'moss'
        }
        
        for code, name in landcover_mapping.items():
            df[f'lc_{name}'] = (df['landcover'] == code).astype(int)
        
        logger.info(f"Land cover distribution:\n{df['landcover'].value_counts().head()}")
        
        return df
    
    def calculate_terrain_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived terrain metrics
        
        Args:
            df: DataFrame with elevation and slope
            
        Returns:
            DataFrame with derived terrain features
        """
        df = df.copy()
        
        if 'elevation' in df.columns and 'slope' in df.columns:
            logger.info("Calculating derived terrain metrics")
            
            # Terrain ruggedness
            df['terrain_ruggedness'] = df['slope'] * (df['elevation'] / (df['elevation'].max() + 1))
            
            # Elevation category
            df['elevation_category'] = pd.cut(
                df['elevation'], 
                bins=5, 
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            
            # Slope category
            df['slope_category'] = pd.cut(
                df['slope'],
                bins=[0, 2, 5, 10, 20, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
        
        return df
    
    def extract_all_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all geospatial features
        
        Args:
            df: DataFrame with Latitude and Longitude columns
            
        Returns:
            DataFrame with all geospatial features
        """
        logger.info("Starting geospatial feature extraction")
        
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            logger.error("DataFrame must contain Latitude and Longitude columns")
            return df
        
        # Extract elevation
        df = self.extract_elevation(df)
        
        # Extract slope
        df = self.extract_slope(df)
        
        # Extract land cover
        df = self.extract_landcover(df)
        
        # Calculate derived metrics
        df = self.calculate_terrain_metrics(df)
        
        logger.info(f"Geospatial extraction complete. Total columns: {len(df.columns)}")
        
        return df


def create_mock_geospatial_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Create mock geospatial features when raster data is not available
    For development and testing purposes
    
    Args:
        df: DataFrame with Latitude and Longitude
        seed: Random seed
        
    Returns:
        DataFrame with mock geospatial features
    """
    logger.warning("Creating mock geospatial features (raster data not available)")
    
    df = df.copy()
    np.random.seed(seed)
    
    # Mock elevation based on latitude (rough approximation)
    df['elevation'] = 100 + (df['Latitude'] - df['Latitude'].min()) * 10 + \
                      np.random.normal(0, 20, len(df))
    
    # Mock slope
    df['slope'] = np.abs(np.random.gamma(2, 2, len(df)))
    
    # Mock land cover (random classes)
    df['landcover'] = np.random.choice([10, 20, 30, 40, 80, 90], len(df))
    
    # One-hot encode
    for code, name in [(10, 'forest'), (20, 'shrubland'), (30, 'grassland'),
                       (40, 'cropland'), (80, 'water'), (90, 'wetland')]:
        df[f'lc_{name}'] = (df['landcover'] == code).astype(int)
    
    # Terrain metrics
    df['terrain_ruggedness'] = df['slope'] * (df['elevation'] / df['elevation'].max())
    df['elevation_category'] = pd.cut(df['elevation'], bins=5, labels=[0,1,2,3,4]).astype(int)
    df['slope_category'] = pd.cut(df['slope'], bins=5, labels=[0,1,2,3,4]).astype(int)
    
    return df
