"""
Configuration loader for alpha-canvas.

This module handles loading and managing YAML configuration files from the config/ directory.
"""

import yaml
from pathlib import Path
from typing import Dict, List


class ConfigLoader:
    """Load and manage YAML configuration files.
    
    The ConfigLoader reads configuration files from the specified directory and provides
    methods to access field definitions and other settings.
    
    Attributes:
        config_dir: Path to the configuration directory
        data_config: Dictionary containing all data field definitions from data.yaml
    
    Example:
        >>> loader = ConfigLoader(config_dir='config')
        >>> fields = loader.list_fields()
        >>> adj_close_def = loader.get_field('adj_close')
        >>> print(adj_close_def['table'])  # 'PRICEVOLUME'
    """
    
    def __init__(self, config_dir: str = 'config'):
        """Initialize ConfigLoader with specified configuration directory.
        
        Args:
            config_dir: Path to directory containing YAML config files (default: 'config')
        """
        self.config_dir = Path(config_dir)
        self.data_config: Dict = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all YAML files from config directory.
        
        Currently loads:
        - data.yaml: Data field definitions
        
        Raises:
            FileNotFoundError: If config directory or required files don't exist
        """
        data_yaml = self.config_dir / 'data.yaml'
        if data_yaml.exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                self.data_config = yaml.safe_load(f)
        else:
            # Initialize empty config if file doesn't exist
            self.data_config = {}
    
    def get_field(self, field_name: str) -> Dict:
        """Get configuration for a specific data field.
        
        Args:
            field_name: Name of the field to retrieve (e.g., 'adj_close')
        
        Returns:
            Dictionary containing field configuration with keys:
            - table: Database table name
            - index_col: Time index column name
            - security_col: Security identifier column name
            - value_col: Value column name
            - query: SQL query string
        
        Raises:
            KeyError: If field_name is not found in configuration
        
        Example:
            >>> loader = ConfigLoader()
            >>> field_def = loader.get_field('adj_close')
            >>> print(field_def['table'])  # 'PRICEVOLUME'
        """
        if field_name not in self.data_config:
            raise KeyError(f"Field '{field_name}' not found in config")
        return self.data_config[field_name]
    
    def list_fields(self) -> List[str]:
        """List all configured field names.
        
        Returns:
            List of field names available in data configuration
        
        Example:
            >>> loader = ConfigLoader()
            >>> fields = loader.list_fields()
            >>> print(fields)  # ['adj_close', 'volume', 'market_cap', ...]
        """
        return list(self.data_config.keys())


