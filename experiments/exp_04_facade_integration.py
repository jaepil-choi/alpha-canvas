"""
Experiment 04: AlphaCanvas Facade Integration

Date: 2025-01-20
Status: In Progress

Objective:
- Validate that all components (Config, DataPanel, Expression, Visitor) integrate correctly

Hypothesis:
- AlphaCanvas facade can initialize all subsystems
- Field expressions can reference data
- add_data() works with both Expression and DataArray

Success Criteria:
- [ ] AlphaCanvas initializes with ConfigLoader
- [ ] DataPanel initialized internally
- [ ] Field('returns') expression works (with mock data)
- [ ] add_data() accepts DataArray (inject pattern)
- [ ] db property returns pure Dataset (eject pattern)
"""

import numpy as np
import pandas as pd
import xarray as xr


# Import from our implementation
from alpha_canvas.core.config import ConfigLoader
from alpha_canvas.core.data_model import DataPanel
from alpha_canvas.core.expression import Expression, Field
from alpha_canvas.core.visitor import EvaluateVisitor


# Minimal AlphaCanvas facade for experiment
class AlphaCanvas:
    """Minimal facade for MVP integration test."""
    
    def __init__(self, config_dir='config', time_index=None, asset_index=None):
        # Load configurations
        self._config = ConfigLoader(config_dir)
        
        # Initialize data panel
        if time_index is None:
            time_index = pd.date_range('2020-01-01', periods=100)
        if asset_index is None:
            asset_index = [f'ASSET_{i}' for i in range(50)]
        
        self._panel = DataPanel(time_index, asset_index)
        self._evaluator = EvaluateVisitor(self._panel.db)
        self.rules = {}
    
    @property
    def db(self):
        """Eject: Return pure xarray.Dataset."""
        return self._panel.db
    
    def add_data(self, name: str, data):
        """Add data (DataArray or Expression)."""
        if isinstance(data, Expression):
            # Store rule and evaluate
            self.rules[name] = data
            result = self._evaluator.evaluate(data)
            self._panel.add_data(name, result)
            # Re-sync evaluator with updated dataset
            self._evaluator = EvaluateVisitor(self._panel.db)
        else:
            # Direct inject
            self._panel.add_data(name, data)
            # Re-sync evaluator
            self._evaluator = EvaluateVisitor(self._panel.db)


def main():
    print("=" * 60)
    print("EXPERIMENT 04: Facade Integration")
    print("=" * 60)
    
    # Step 1: Create AlphaCanvas
    print("\n[Step 1] Creating AlphaCanvas...")
    try:
        rc = AlphaCanvas(config_dir='config')
        print(f"  [OK] rc = AlphaCanvas(config_dir='config')")
        print(f"  [OK] ConfigLoader initialized")
        print(f"  [OK] DataPanel initialized (empty)")
        print(f"  [OK] rc.db type: {type(rc.db)}")
    except Exception as e:
        print(f"  [FAIL] Error creating AlphaCanvas: {e}")
        return
    
    # Step 2: Test config access
    print("\n[Step 2] Testing config access...")
    try:
        fields = rc._config.list_fields()
        print(f"  [OK] Config has {len(fields)} fields: {', '.join(fields[:3])}...")
        
        adj_close_def = rc._config.get_field('adj_close')
        print(f"  [OK] adj_close definition loaded")
        print(f"    table: {adj_close_def['table']}")
    except Exception as e:
        print(f"  [FAIL] Config access error: {e}")
        return
    
    # Step 3: Add data directly (inject pattern)
    print("\n[Step 3] Adding data directly (inject pattern)...")
    try:
        time_idx = pd.date_range('2020-01-01', periods=100)
        asset_idx = [f'ASSET_{i}' for i in range(50)]
        
        # Create mock returns data
        returns_data = xr.DataArray(
            np.random.randn(100, 50) * 0.02,
            dims=['time', 'asset'],
            coords={'time': time_idx, 'asset': asset_idx}
        )
        
        rc.add_data('returns', returns_data)
        print(f"  [OK] Added 'returns' DataArray")
        print(f"    Shape: {rc.db['returns'].shape}")
        print(f"    Type: {rc.db['returns'].dtype}")
    except Exception as e:
        print(f"  [FAIL] Error adding data: {e}")
        return
    
    # Step 4: Test Field expression
    print("\n[Step 4] Testing Field expression...")
    try:
        # Create Field expression
        field_expr = Field('returns')
        print(f"  [OK] Created Field('returns') expression")
        
        # Evaluate it
        result = rc._evaluator.evaluate(field_expr)
        print(f"  [OK] Evaluated Field expression")
        print(f"    Result shape: {result.shape}")
        print(f"    Result matches rc.db['returns']: {np.array_equal(result.values, rc.db['returns'].values)}")
    except Exception as e:
        print(f"  [FAIL] Error with Field expression: {e}")
        return
    
    # Step 5: Test add_data with Expression
    print("\n[Step 5] Testing add_data with Expression...")
    try:
        # Add market_cap data first
        mcap_data = xr.DataArray(
            np.random.randn(100, 50) * 1000 + 5000,
            dims=['time', 'asset'],
            coords={'time': time_idx, 'asset': asset_idx}
        )
        rc.add_data('mcap', mcap_data)
        print(f"  [OK] Added 'mcap' DataArray")
        
        # Now add via Expression
        mcap_field_expr = Field('mcap')
        rc.add_data('mcap_copy', mcap_field_expr)
        print(f"  [OK] Added 'mcap_copy' via Field Expression")
        print(f"    Expression stored in rules: {'mcap_copy' in rc.rules}")
        print(f"    Data accessible: {'mcap_copy' in rc.db.data_vars}")
    except Exception as e:
        print(f"  [FAIL] Error with Expression add_data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Test eject pattern
    print("\n[Step 6] Testing eject pattern...")
    try:
        pure_ds = rc.db
        print(f"  [OK] Ejected dataset type: {type(pure_ds)}")
        print(f"  [OK] Is pure xarray.Dataset: {type(pure_ds) == xr.Dataset}")
        print(f"  [OK] Data vars: {list(pure_ds.data_vars)}")
    except Exception as e:
        print(f"  [FAIL] Eject error: {e}")
        return
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT: [SUCCESS] - Minimal facade integration works")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Config: {len(rc._config.list_fields())} fields loaded")
    print(f"  - DataPanel: {len(rc.db.data_vars)} data variables")
    print(f"  - Rules: {len(rc.rules)} expression rules")
    print(f"  - Dataset shape: {dict(rc.db.sizes)}")


if __name__ == '__main__':
    main()


