"""
alpha-database: Data persistence and access layer for alpha-canvas ecosystem.

This package provides:
- Config-driven data loading via DataSource
- Multi-backend support (Parquet, CSV, PostgreSQL, etc.)
- Stateless design (date ranges passed per call)
- Plugin architecture for custom readers

Phase 1 (Current): Config-Driven Data Loading
- DataSource facade for loading fields
- BaseReader interface for custom readers
- ParquetReader for file-based data (MVP)

Phase 2 (Future): Data Writing & Catalogs
- Dataset Catalog: Store computed fields with schema evolution
- Alpha Catalog: Version-controlled alpha signal storage
- Factor Catalog: Time-series factor returns storage

Design Principles:
- Loosely coupled with alpha-canvas via public APIs
- Pluggable backends for storage flexibility
- Stateless and reusable components
- Config-driven for flexibility

Usage:
    from alpha_database import DataSource, BaseReader
    
    # Load data fields
    ds = DataSource('config/data.yaml')
    adj_close = ds.load_field('adj_close', '2024-01-01', '2024-12-31')
    
    # Register custom reader
    ds.register_reader('custom', CustomReader())
    custom_data = ds.load_field('custom_field', '2024-01-01', '2024-12-31')
"""

from .core.data_source import DataSource
from .readers.base import BaseReader

__version__ = "0.1.0"

__all__ = [
    'DataSource',
    'BaseReader',
]

