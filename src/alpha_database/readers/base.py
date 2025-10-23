"""
Base reader interface for alpha-database.

This module defines the abstract base class for all data readers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


class BaseReader(ABC):
    """Abstract base class for data readers.
    
    All readers (ParquetReader, CSVReader, PostgresReader, etc.) must implement
    this interface. The `read` method takes a query string and parameters,
    and returns a long-format DataFrame.
    
    Example:
        >>> class CustomReader(BaseReader):
        ...     def read(self, query: str, params: Dict[str, Any]) -> pd.DataFrame:
        ...         # Custom implementation
        ...         return df
    """
    
    @abstractmethod
    def read(self, query: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Read data and return long-format DataFrame.
        
        Args:
            query: Query string (format depends on reader implementation)
                   For file-based readers: SQL query for DuckDB
                   For DB readers: Native SQL query
            params: Query parameters (e.g., {'start_date': '2024-01-01', 'end_date': '2024-12-31'})
        
        Returns:
            Long-format DataFrame with at least these columns:
            - Time column (date/datetime)
            - Asset column (security identifier)
            - Value column (numeric data)
        
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        
        Example:
            >>> reader = ParquetReader()
            >>> df = reader.read(
            ...     query="SELECT * FROM parquet_file WHERE date >= :start_date",
            ...     params={'start_date': '2024-01-01', 'end_date': '2024-12-31'}
            ... )
        """
        pass

