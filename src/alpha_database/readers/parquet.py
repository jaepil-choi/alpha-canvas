"""
Parquet reader for alpha-database.

This module provides a reader for Parquet files using DuckDB for querying.
"""

from typing import Any, Dict
import pandas as pd
import duckdb
from .base import BaseReader


class ParquetReader(BaseReader):
    """Reader for Parquet files using DuckDB.
    
    This reader executes SQL queries against Parquet files using DuckDB's
    in-memory SQL engine. It supports parameter substitution for dynamic queries.
    
    Example:
        >>> reader = ParquetReader()
        >>> df = reader.read(
        ...     query="SELECT * FROM 'data/prices.parquet' WHERE date >= :start_date",
        ...     params={'start_date': '2024-01-01', 'end_date': '2024-12-31'}
        ... )
    """
    
    def __init__(self):
        """Initialize ParquetReader."""
        pass
    
    def read(self, query: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Read Parquet file using DuckDB SQL query.
        
        Args:
            query: SQL query string with parameter placeholders (e.g., :start_date, :end_date)
            params: Dictionary of parameter values to substitute into query
        
        Returns:
            Long-format DataFrame with query results
        
        Example:
            >>> query = '''
            ...     SELECT date, ticker, adj_close
            ...     FROM 'data/pricevolume.parquet'
            ...     WHERE date >= :start_date AND date <= :end_date
            ... '''
            >>> params = {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
            >>> df = reader.read(query, params)
        
        Notes:
            - Parameter placeholders use :param_name syntax
            - String parameters are automatically quoted
            - Query is executed using DuckDB's in-memory engine
            - DuckDB auto-detects Parquet schema and optimizes queries
        """
        # Substitute parameters into query
        # DuckDB doesn't support native parameterized queries for file paths,
        # so we do string substitution. For security, this should only be used
        # with trusted config files, not user input.
        formatted_query = query
        for param_name, param_value in params.items():
            placeholder = f":{param_name}"
            # Quote string values
            if isinstance(param_value, str):
                formatted_value = f"'{param_value}'"
            else:
                formatted_value = str(param_value)
            formatted_query = formatted_query.replace(placeholder, formatted_value)
        
        # Execute query with DuckDB
        result = duckdb.query(formatted_query).to_df()
        
        return result


