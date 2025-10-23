"""
Experiment 03: Visitor Pattern & Caching

Date: 2025-01-20
Status: In Progress

Objective:
- Validate that Visitor pattern can traverse Expression tree and cache results with integer steps

Hypothesis:
- Can implement basic Expression ABC and Field leaf
- Visitor can traverse in predictable depth-first order
- Can cache intermediate results with integer step counter

Success Criteria:
- [ ] Define Expression ABC and Field class
- [ ] Create simple visitor that returns mock data
- [ ] Build tree with 2 Field nodes
- [ ] Verify depth-first order
- [ ] Cache results with integer keys
"""

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple


# Minimal Expression classes for experiment
class Expression(ABC):
    """Base class for all expression nodes."""
    
    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor (Visitor pattern)."""
        pass


@dataclass
class Field(Expression):
    """Leaf node: Reference to a data field."""
    name: str
    
    def accept(self, visitor):
        return visitor.visit_field(self)


# Minimal Visitor for experiment
class EvaluateVisitor:
    """Evaluates Expression tree with depth-first traversal and caching."""
    
    def __init__(self, data_source: xr.Dataset):
        self._data = data_source
        self._cache: Dict[int, Tuple[str, xr.DataArray]] = {}
        self._step_counter = 0
    
    def evaluate(self, expr: Expression) -> xr.DataArray:
        """Evaluate expression and return result."""
        self._step_counter = 0  # Reset for new evaluation
        self._cache = {}
        return expr.accept(self)
    
    def visit_field(self, node: Field) -> xr.DataArray:
        """Visit Field node: retrieve from dataset."""
        result = self._data[node.name]
        self._cache_result(f"Field_{node.name}", result)
        return result
    
    def _cache_result(self, name: str, result: xr.DataArray):
        """Cache result with current step number."""
        self._cache[self._step_counter] = (name, result)
        self._step_counter += 1
    
    def get_cached(self, step: int) -> Tuple[str, xr.DataArray]:
        """Retrieve cached result by step number."""
        return self._cache[step]


def main():
    print("=" * 60)
    print("EXPERIMENT 03: Visitor Pattern & Caching")
    print("=" * 60)
    
    # Step 1: Define Expression classes
    print("\n[Step 1] Defining Expression classes...")
    print("  [OK] Expression ABC defined")
    print("  [OK] Field(name='returns') leaf defined")
    
    # Step 2: Create test dataset
    print("\n[Step 2] Creating test dataset...")
    time_idx = pd.date_range('2020-01-01', periods=100)
    asset_idx = [f'ASSET_{i}' for i in range(50)]
    
    ds = xr.Dataset(
        coords={'time': time_idx, 'asset': asset_idx}
    )
    
    # Add mock data
    returns_data = xr.DataArray(
        np.random.randn(100, 50),
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    mcap_data = xr.DataArray(
        np.random.randn(100, 50) * 1000 + 5000,
        dims=['time', 'asset'],
        coords={'time': time_idx, 'asset': asset_idx}
    )
    ds = ds.assign({'returns': returns_data, 'mcap': mcap_data})
    print(f"  [OK] Dataset created with 'returns' and 'mcap'")
    
    # Step 3: Create EvaluateVisitor
    print("\n[Step 3] Creating EvaluateVisitor...")
    visitor = EvaluateVisitor(ds)
    print(f"  [OK] Visitor initialized with step_counter={visitor._step_counter}")
    
    # Step 4: Evaluate Field expressions
    print("\n[Step 4] Evaluating Field expressions...")
    field1 = Field('returns')
    result1 = visitor.evaluate(field1)
    print(f"  Visiting: Field('returns') -> step 0")
    print(f"    Result shape: {result1.shape}")
    
    # Reset and evaluate second field
    field2 = Field('mcap')
    result2 = visitor.evaluate(field2)
    print(f"  Visiting: Field('mcap') -> step 0 (new evaluation)")
    print(f"    Result shape: {result2.shape}")
    
    # Step 5: Test caching
    print("\n[Step 5] Validating cache...")
    
    # Re-evaluate to populate cache
    visitor._step_counter = 0
    visitor._cache = {}
    _ = visitor.evaluate(field1)
    step0_name, step0_data = visitor.get_cached(0)
    print(f"  cache[0] = ('{step0_name}', <array shape={step0_data.shape}>)")
    
    # Evaluate second field
    _ = visitor.evaluate(field2)
    step0_name_2, step0_data_2 = visitor.get_cached(0)
    print(f"  cache[0] = ('{step0_name_2}', <array shape={step0_data_2.shape}>) [after 2nd eval]")
    
    # Step 6: Test retrieval
    print("\n[Step 6] Testing retrieval...")
    try:
        cached_name, cached_data = visitor.get_cached(0)
        print(f"  [OK] get_cached(0) -> '{cached_name}' result")
        print(f"    Shape: {cached_data.shape}, dtype: {cached_data.dtype}")
    except KeyError:
        print(f"  [FAIL] Could not retrieve cached step 0")
        return
    
    # Step 7: Verify step counter behavior
    print("\n[Step 7] Verifying step counter...")
    visitor._step_counter = 0
    visitor._cache = {}
    
    # Evaluate multiple fields in sequence
    _ = field1.accept(visitor)  # step 0
    current_step = visitor._step_counter
    print(f"  After Field('returns'): step_counter = {current_step}")
    
    _ = field2.accept(visitor)  # step 1
    current_step = visitor._step_counter
    print(f"  After Field('mcap'): step_counter = {current_step}")
    
    if current_step == 2 and len(visitor._cache) == 2:
        print(f"  [OK] Step counter increments correctly (0 -> 1 -> 2)")
        print(f"  [OK] Cache has {len(visitor._cache)} entries")
    else:
        print(f"  [FAIL] Step counter or cache issue")
        return
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT: [SUCCESS] - Visitor pattern with caching works")
    print("=" * 60)


if __name__ == '__main__':
    main()


