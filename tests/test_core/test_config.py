"""
Tests for ConfigLoader class.

These tests follow TDD methodology - they are written before implementation
to define the expected behavior of the ConfigLoader.
"""

import pytest
from pathlib import Path
from alpha_canvas.core.config import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader class."""
    
    def test_config_loader_initialization(self):
        """Test ConfigLoader can be created with config directory."""
        loader = ConfigLoader(config_dir='config')
        assert loader is not None
        assert isinstance(loader.config_dir, Path)
    
    def test_load_data_yaml(self):
        """Test loading data.yaml file."""
        loader = ConfigLoader(config_dir='config')
        assert loader.data_config is not None
        assert isinstance(loader.data_config, dict)
        assert 'adj_close' in loader.data_config
        assert 'returns' in loader.data_config
    
    def test_get_field_definition(self):
        """Test retrieving a specific field definition."""
        loader = ConfigLoader(config_dir='config')
        field_def = loader.get_field('adj_close')
        
        # Validate structure
        assert isinstance(field_def, dict)
        assert field_def['db_type'] == 'parquet'
        assert field_def['table'] == 'PRICEVOLUME'
        assert field_def['index_col'] == 'date'
        assert field_def['security_col'] == 'security_id'
        assert field_def['value_col'] == 'adj_close'
        assert 'query' in field_def
        assert len(field_def['query']) > 0
    
    def test_get_field_missing(self):
        """Test getting a non-existent field raises KeyError."""
        loader = ConfigLoader(config_dir='config')
        with pytest.raises(KeyError) as exc_info:
            loader.get_field('nonexistent_field')
        assert 'nonexistent_field' in str(exc_info.value)
    
    def test_list_available_fields(self):
        """Test listing all configured fields."""
        loader = ConfigLoader(config_dir='config')
        fields = loader.list_fields()
        
        assert isinstance(fields, list)
        assert 'adj_close' in fields
        assert 'market_cap' in fields
        assert 'returns' in fields
        assert 'volume' in fields
        assert 'subindustry' in fields
        assert len(fields) == 5  # Based on current config/data.yaml
    
    def test_config_dir_validation(self):
        """Test that config_dir is stored as Path object."""
        loader = ConfigLoader(config_dir='config')
        assert isinstance(loader.config_dir, Path)
        assert str(loader.config_dir) == 'config'
    
    def test_data_config_structure(self):
        """Test that data_config has correct nested structure."""
        loader = ConfigLoader(config_dir='config')
        
        # Check that all fields have required keys
        required_keys = ['table', 'index_col', 'security_col', 'value_col', 'query']
        for field_name, field_def in loader.data_config.items():
            for key in required_keys:
                assert key in field_def, f"Field '{field_name}' missing key '{key}'"


