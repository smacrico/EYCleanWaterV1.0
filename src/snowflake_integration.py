"""
Snowflake Integration Module for EY Water Quality Challenge
Upload and query data from Snowflake
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    logger.warning("Snowflake connector not available. Install with: pip install snowflake-connector-python")


class SnowflakeClient:
    """Client for Snowflake database operations"""
    
    def __init__(self, config: Dict):
        """
        Initialize Snowflake client
        
        Args:
            config: Configuration dictionary with Snowflake credentials
        """
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("snowflake-connector-python not installed")
        
        self.config = config
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Establish connection to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                user=self.config['user'],
                password=self.config['password'],
                account=self.config['account'],
                warehouse=self.config.get('warehouse', 'COMPUTE_WH'),
                database=self.config.get('database'),
                schema=self.config.get('schema')
            )
            self.cursor = self.connection.cursor()
            logger.info("Connected to Snowflake successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
    
    def disconnect(self):
        """Close Snowflake connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Disconnected from Snowflake")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        if not self.connection:
            self.connect()
        
        try:
            logger.info(f"Executing query: {query[:100]}...")
            self.cursor.execute(query)
            
            # Fetch results
            columns = [desc[0] for desc in self.cursor.description]
            data = self.cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            logger.info(f"Query returned {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def upload_dataframe(self, 
                        df: pd.DataFrame,
                        table_name: str,
                        database: Optional[str] = None,
                        schema: Optional[str] = None,
                        if_exists: str = 'replace') -> bool:
        """
        Upload pandas DataFrame to Snowflake table
        
        Args:
            df: DataFrame to upload
            table_name: Target table name
            database: Database name (optional, uses default from config)
            schema: Schema name (optional, uses default from config)
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            
        Returns:
            True if successful
        """
        if not self.connection:
            self.connect()
        
        database = database or self.config.get('database')
        schema = schema or self.config.get('schema')
        
        try:
            logger.info(f"Uploading {len(df)} rows to {database}.{schema}.{table_name}")
            
            # Use Snowflake's optimized write function
            success, num_chunks, num_rows, output = write_pandas(
                conn=self.connection,
                df=df,
                table_name=table_name,
                database=database,
                schema=schema,
                auto_create_table=True,
                overwrite=(if_exists == 'replace')
            )
            
            if success:
                logger.info(f"Successfully uploaded {num_rows} rows in {num_chunks} chunks")
                return True
            else:
                logger.error("Upload failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {e}")
            raise
    
    def create_table_from_dataframe(self, 
                                   df: pd.DataFrame,
                                   table_name: str,
                                   database: Optional[str] = None,
                                   schema: Optional[str] = None) -> bool:
        """
        Create Snowflake table with schema matching DataFrame
        
        Args:
            df: DataFrame to use for schema
            table_name: Name of table to create
            database: Database name
            schema: Schema name
            
        Returns:
            True if successful
        """
        database = database or self.config.get('database')
        schema = schema or self.config.get('schema')
        
        # Map pandas dtypes to Snowflake types
        dtype_mapping = {
            'int64': 'INTEGER',
            'float64': 'FLOAT',
            'object': 'VARCHAR',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP'
        }
        
        # Build CREATE TABLE statement
        columns = []
        for col, dtype in df.dtypes.items():
            sf_type = dtype_mapping.get(str(dtype), 'VARCHAR')
            columns.append(f'"{col}" {sf_type}')
        
        create_stmt = f"""
        CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
            {', '.join(columns)}
        )
        """
        
        try:
            logger.info(f"Creating table {table_name}")
            self.cursor.execute(create_stmt)
            logger.info(f"Table {table_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    def load_training_data(self, table_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data from Snowflake
        
        Args:
            table_name: Table name (optional)
            
        Returns:
            DataFrame with training data
        """
        if table_name is None:
            table_name = f"{self.config.get('table_prefix', 'WQ_')}TRAINING"
        
        query = f"SELECT * FROM {table_name}"
        return self.execute_query(query)
    
    def load_validation_data(self, table_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load validation data from Snowflake
        
        Args:
            table_name: Table name (optional)
            
        Returns:
            DataFrame with validation data
        """
        if table_name is None:
            table_name = f"{self.config.get('table_prefix', 'WQ_')}VALIDATION"
        
        query = f"SELECT * FROM {table_name}"
        return self.execute_query(query)
    
    def save_predictions(self, 
                        predictions_df: pd.DataFrame,
                        table_name: Optional[str] = None) -> bool:
        """
        Save predictions to Snowflake
        
        Args:
            predictions_df: DataFrame with predictions
            table_name: Target table name
            
        Returns:
            True if successful
        """
        if table_name is None:
            table_name = f"{self.config.get('table_prefix', 'WQ_')}PREDICTIONS"
        
        return self.upload_dataframe(
            predictions_df,
            table_name,
            if_exists='replace'
        )
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """
        Get information about a table
        
        Args:
            table_name: Name of table
            
        Returns:
            DataFrame with table schema information
        """
        query = f"DESCRIBE TABLE {table_name}"
        return self.execute_query(query)


def get_snowflake_client(config: Dict) -> Optional[SnowflakeClient]:
    """
    Get Snowflake client if enabled in config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SnowflakeClient instance or None if disabled
    """
    snowflake_config = config.get('snowflake', {})
    
    if not snowflake_config.get('enabled', False):
        logger.info("Snowflake integration disabled in config")
        return None
    
    if not SNOWFLAKE_AVAILABLE:
        logger.warning("Snowflake connector not installed")
        return None
    
    try:
        client = SnowflakeClient(snowflake_config)
        client.connect()
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Snowflake client: {e}")
        return None


# Example usage functions
def upload_competition_data_to_snowflake(config: Dict, 
                                        train_df: pd.DataFrame,
                                        val_df: pd.DataFrame):
    """
    Upload competition data to Snowflake
    
    Args:
        config: Configuration dictionary
        train_df: Training DataFrame
        val_df: Validation DataFrame
    """
    client = get_snowflake_client(config)
    
    if client is None:
        logger.info("Skipping Snowflake upload")
        return
    
    try:
        # Upload training data
        client.upload_dataframe(
            train_df,
            f"{config['snowflake']['table_prefix']}TRAINING",
            if_exists='replace'
        )
        
        # Upload validation data
        client.upload_dataframe(
            val_df,
            f"{config['snowflake']['table_prefix']}VALIDATION",
            if_exists='replace'
        )
        
        logger.info("Successfully uploaded all data to Snowflake")
        
    finally:
        client.disconnect()


def load_competition_data_from_snowflake(config: Dict) -> tuple:
    """
    Load competition data from Snowflake
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, val_df)
    """
    client = get_snowflake_client(config)
    
    if client is None:
        raise ValueError("Snowflake not configured or not available")
    
    try:
        train_df = client.load_training_data()
        val_df = client.load_validation_data()
        
        return train_df, val_df
        
    finally:
        client.disconnect()
