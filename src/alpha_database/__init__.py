"""
Alpha-Database: Persistence Layer for Alpha Signals and Datasets

This package provides data persistence capabilities for alpha-canvas,
including dataset catalogs, alpha signal storage, and factor time series.

Core Features:
- Dataset Catalog (D1): Store and manage computed fields with schema evolution
- Alpha Catalog (D2): Version-controlled alpha signal storage
- Factor Catalog (D3): Time-series factor returns storage

Storage Backends:
- Parquet (MVP): File-based storage
- PostgreSQL (Future): Relational database backend
- ClickHouse (Future): Columnar OLAP backend

Design Principles:
- Loosely coupled with alpha-canvas via public APIs
- Pluggable backends for storage flexibility
- Schema evolution on-the-fly
- Versioning for reproducibility

Usage:
    from alpha_database import DatasetCatalog, AlphaCatalog, FactorCatalog
    
    # Save computed field to dataset
    catalog = DatasetCatalog(backend='parquet', path='./data')
    catalog.save_field('fundamental', 'pbr', data)
    
    # Save alpha signal
    alpha_catalog = AlphaCatalog(backend='parquet', path='./alphas')
    alpha_catalog.save_alpha(
        alpha_id='momentum_v1',
        signal=signal_data,
        weights=weight_data,
        returns=return_data,
        metadata={'author': 'researcher', 'date': '2025-01-01'}
    )
"""

__version__ = "0.1.0"

# Public API placeholder - will be implemented later
__all__ = [
    # "DatasetCatalog",  # TODO: Implement in Phase 1
    # "AlphaCatalog",    # TODO: Implement in Phase 2
    # "FactorCatalog",   # TODO: Implement in Phase 3
]

