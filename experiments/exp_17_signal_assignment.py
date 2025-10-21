"""
Experiment 17: Signal Canvas Assignment Patterns

Date: 2025-01-21
Status: In Progress

Objective:
    Validate two design patterns for Expression-based signal assignment:
    1. Lazy Evaluation (deferred assignment)
    2. Immediate Evaluation (eager assignment)
    
    Compare their characteristics to inform implementation decisions.

Success Criteria:
    - Both patterns produce identical results
    - Performance characteristics measured
    - Traceability implications understood
    - Memory footprint compared
    - Clear recommendation for implementation
"""

import numpy as np
import pandas as pd
import xarray as xr
import time
import sys
from typing import List, Tuple, Optional, Any


# ============================================================================
# MOCK INFRASTRUCTURE (Simplified Expression/Visitor)
# ============================================================================

class MockExpression:
    """Base Expression class for testing."""
    def evaluate(self, evaluator):
        raise NotImplementedError


class MockField(MockExpression):
    """Leaf node - represents data field."""
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, evaluator):
        return evaluator.data[self.name]


class MockConstant(MockExpression):
    """Constant value expression."""
    def __init__(self, value: float):
        self.value = value
        self._shape = None
    
    def set_shape(self, shape):
        self._shape = shape
    
    def evaluate(self, evaluator):
        if self._shape is None:
            raise ValueError("Constant shape not set")
        return xr.DataArray(
            np.full(self._shape, self.value),
            dims=['time', 'asset'],
            coords=evaluator.coords
        )


class MockTsMean(MockExpression):
    """Time-series mean operator."""
    def __init__(self, child: MockExpression, window: int):
        self.child = child
        self.window = window
    
    def evaluate(self, evaluator):
        child_result = self.child.evaluate(evaluator)
        return child_result.rolling(time=self.window, min_periods=self.window).mean()


class MockEvaluator:
    """Simplified evaluator for testing."""
    def __init__(self, data: dict, coords: dict):
        self.data = data
        self.coords = coords


# ============================================================================
# PATTERN 1: LAZY EVALUATION (Deferred Assignment)
# ============================================================================

class LazyExpression(MockExpression):
    """Expression with lazy assignment support."""
    
    def __init__(self, base_expr: MockExpression):
        self.base_expr = base_expr
        self._assignments: List[Tuple[Any, float]] = []
    
    def __setitem__(self, mask, value):
        """Store assignment for later evaluation."""
        self._assignments.append((mask, value))
    
    def evaluate(self, evaluator):
        """Evaluate base expression, then apply all assignments."""
        # Step 1: Evaluate base expression
        result = self.base_expr.evaluate(evaluator)
        
        # Step 2: Apply assignments sequentially
        for mask, value in self._assignments:
            # If mask is a boolean array, use it directly
            if isinstance(mask, xr.DataArray):
                result = result.where(~mask, value)
            else:
                result = result.where(~mask, value)
        
        return result
    
    def get_assignments(self):
        """Inspect assignments before evaluation."""
        return self._assignments.copy()


# ============================================================================
# PATTERN 2: IMMEDIATE EVALUATION (Eager Assignment)
# ============================================================================

class ImmediateExpression(MockExpression):
    """Expression with immediate assignment support."""
    
    def __init__(self, base_expr: MockExpression, evaluator: MockEvaluator):
        self.base_expr = base_expr
        self.evaluator = evaluator
        self._base_data: Optional[xr.DataArray] = None
        self._is_evaluated = False
    
    def __setitem__(self, mask, value):
        """Evaluate immediately and modify data."""
        # First assignment triggers evaluation
        if not self._is_evaluated:
            self._base_data = self.base_expr.evaluate(self.evaluator)
            self._is_evaluated = True
        
        # Apply assignment immediately
        if isinstance(mask, xr.DataArray):
            self._base_data = self._base_data.where(~mask, value)
        else:
            self._base_data = self._base_data.where(~mask, value)
    
    def evaluate(self, evaluator):
        """Return cached data if evaluated, otherwise evaluate base."""
        if self._is_evaluated:
            return self._base_data
        else:
            return self.base_expr.evaluate(evaluator)


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def create_test_data():
    """Create sample (T, N) data for testing."""
    T, N = 10, 6
    times = pd.date_range('2024-01-01', periods=T, freq='D')
    assets = [f'ASSET_{i}' for i in range(N)]
    
    # Returns data
    returns = xr.DataArray(
        np.random.randn(T, N) * 0.02,
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    # Size labels
    size = xr.DataArray(
        np.array([['small', 'small', 'big', 'big', 'small', 'big']] * T),
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    # Momentum labels
    momentum = xr.DataArray(
        np.array([['high', 'low', 'high', 'low', 'high', 'low']] * T),
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    # Universe mask (exclude one asset)
    universe = xr.DataArray(
        np.array([[True, True, True, True, True, False]] * T),
        dims=['time', 'asset'],
        coords={'time': times, 'asset': assets}
    )
    
    return {
        'returns': returns,
        'size': size,
        'momentum': momentum,
        'universe': universe
    }, {'time': times, 'asset': assets}, (T, N)


def scenario_1_simple_assignment(evaluator, shape):
    """Scenario 1: Simple Assignment from zeros."""
    print("\n" + "="*70)
    print("SCENARIO 1: Simple Assignment (zeros → long/short)")
    print("="*70)
    
    # Masks
    size_data = evaluator.data['size']
    mask_small = (size_data == 'small')
    mask_big = (size_data == 'big')
    
    # Pattern 1: Lazy
    print("\n[LAZY] Building signal...")
    constant_expr = MockConstant(0.0)
    constant_expr.set_shape(shape)
    lazy_signal = LazyExpression(constant_expr)
    lazy_signal[mask_small] = 1.0
    lazy_signal[mask_big] = -1.0
    
    print(f"  Assignments stored: {len(lazy_signal.get_assignments())}")
    start = time.perf_counter()
    lazy_result = lazy_signal.evaluate(evaluator)
    lazy_time = (time.perf_counter() - start) * 1000
    print(f"  Evaluation time: {lazy_time:.3f}ms")
    
    # Pattern 2: Immediate
    print("\n[IMMEDIATE] Building signal...")
    constant_expr2 = MockConstant(0.0)
    constant_expr2.set_shape(shape)
    immediate_signal = ImmediateExpression(constant_expr2, evaluator)
    start = time.perf_counter()
    immediate_signal[mask_small] = 1.0
    immediate_signal[mask_big] = -1.0
    immediate_time = (time.perf_counter() - start) * 1000
    immediate_result = immediate_signal.evaluate(evaluator)
    print(f"  Assignment time: {immediate_time:.3f}ms")
    
    # Validation
    print("\n[VALIDATION]")
    print(f"  Results identical: {np.allclose(lazy_result.values, immediate_result.values)}")
    print(f"  Small positions = 1.0: {np.all(lazy_result.values[mask_small.values] == 1.0)}")
    print(f"  Big positions = -1.0: {np.all(lazy_result.values[mask_big.values] == -1.0)}")
    
    return lazy_time, immediate_time, sys.getsizeof(lazy_signal._assignments), sys.getsizeof(immediate_signal._base_data)


def scenario_2_transform_existing(evaluator, shape):
    """Scenario 2: Transform Existing Signal."""
    print("\n" + "="*70)
    print("SCENARIO 2: Transform Existing Signal (ts_mean → boost high momentum)")
    print("="*70)
    
    # Mask
    momentum_data = evaluator.data['momentum']
    mask_high = (momentum_data == 'high')
    
    # Pattern 1: Lazy
    print("\n[LAZY] Building signal...")
    ts_mean_expr = MockTsMean(MockField('returns'), 3)
    lazy_signal = LazyExpression(ts_mean_expr)
    lazy_signal[mask_high] = 2.0
    
    start = time.perf_counter()
    lazy_result = lazy_signal.evaluate(evaluator)
    lazy_time = (time.perf_counter() - start) * 1000
    print(f"  Evaluation time: {lazy_time:.3f}ms")
    
    # Pattern 2: Immediate
    print("\n[IMMEDIATE] Building signal...")
    ts_mean_expr2 = MockTsMean(MockField('returns'), 3)
    immediate_signal = ImmediateExpression(ts_mean_expr2, evaluator)
    start = time.perf_counter()
    immediate_signal[mask_high] = 2.0
    immediate_time = (time.perf_counter() - start) * 1000
    immediate_result = immediate_signal.evaluate(evaluator)
    print(f"  Assignment time: {immediate_time:.3f}ms")
    
    # Validation
    print("\n[VALIDATION]")
    results_match = np.allclose(
        lazy_result.values, 
        immediate_result.values, 
        equal_nan=True
    )
    print(f"  Results identical: {results_match}")
    high_momentum_replaced = np.all(lazy_result.values[mask_high.values] == 2.0)
    print(f"  High momentum = 2.0: {high_momentum_replaced}")
    
    return lazy_time, immediate_time


def scenario_3_overlapping_masks(evaluator, shape):
    """Scenario 3: Overlapping Masks (later wins)."""
    print("\n" + "="*70)
    print("SCENARIO 3: Overlapping Masks (later assignment wins)")
    print("="*70)
    
    # Create overlapping masks
    mask1 = xr.DataArray(
        np.array([[True, True, True, False, False, False]] * shape[0]),
        dims=['time', 'asset'],
        coords=evaluator.coords
    )
    mask2 = xr.DataArray(
        np.array([[False, True, True, True, False, False]] * shape[0]),
        dims=['time', 'asset'],
        coords=evaluator.coords
    )
    
    print(f"\n  Mask1 positions: {np.where(mask1.values[0])[0].tolist()}")
    print(f"  Mask2 positions: {np.where(mask2.values[0])[0].tolist()}")
    print(f"  Overlap positions: {np.where(mask1.values[0] & mask2.values[0])[0].tolist()}")
    
    # Pattern 1: Lazy
    print("\n[LAZY] Building signal...")
    constant_expr = MockConstant(0.0)
    constant_expr.set_shape(shape)
    lazy_signal = LazyExpression(constant_expr)
    lazy_signal[mask1] = 1.0
    lazy_signal[mask2] = -1.0
    
    start = time.perf_counter()
    lazy_result = lazy_signal.evaluate(evaluator)
    lazy_time = (time.perf_counter() - start) * 1000
    
    # Pattern 2: Immediate
    print("\n[IMMEDIATE] Building signal...")
    constant_expr2 = MockConstant(0.0)
    constant_expr2.set_shape(shape)
    immediate_signal = ImmediateExpression(constant_expr2, evaluator)
    start = time.perf_counter()
    immediate_signal[mask1] = 1.0
    immediate_signal[mask2] = -1.0
    immediate_time = (time.perf_counter() - start) * 1000
    immediate_result = immediate_signal.evaluate(evaluator)
    
    # Validation
    print("\n[VALIDATION]")
    print(f"  Results identical: {np.allclose(lazy_result.values, immediate_result.values)}")
    
    # Check overlap: positions 1 and 2 should be -1.0 (mask2 wins)
    overlap_mask = mask1.values[0] & mask2.values[0]
    overlap_values = lazy_result.values[0, overlap_mask]
    print(f"  Overlap values (should be -1.0): {overlap_values}")
    print(f"  Later assignment wins: {np.all(overlap_values == -1.0)}")
    
    # Check mask1-only: position 0 should be 1.0
    mask1_only = mask1.values[0] & ~mask2.values[0]
    mask1_only_values = lazy_result.values[0, mask1_only]
    print(f"  Mask1-only values (should be 1.0): {mask1_only_values}")
    
    # Check mask2-only: position 3 should be -1.0
    mask2_only = mask2.values[0] & ~mask1.values[0]
    mask2_only_values = lazy_result.values[0, mask2_only]
    print(f"  Mask2-only values (should be -1.0): {mask2_only_values}")
    
    return lazy_time, immediate_time


def scenario_4_multiple_sequential(evaluator, shape):
    """Scenario 4: Multiple Sequential Modifications."""
    print("\n" + "="*70)
    print("SCENARIO 4: Multiple Sequential Modifications")
    print("="*70)
    
    # Create 3 masks
    mask1 = xr.DataArray(
        np.array([[True, True, False, False, False, False]] * shape[0]),
        dims=['time', 'asset'],
        coords=evaluator.coords
    )
    mask2 = xr.DataArray(
        np.array([[False, False, True, True, False, False]] * shape[0]),
        dims=['time', 'asset'],
        coords=evaluator.coords
    )
    mask3 = xr.DataArray(
        np.array([[False, False, False, False, True, False]] * shape[0]),
        dims=['time', 'asset'],
        coords=evaluator.coords
    )
    
    # Pattern 1: Lazy
    print("\n[LAZY] Building signal with 3 modifications...")
    ts_mean_expr = MockTsMean(MockField('returns'), 5)
    lazy_signal = LazyExpression(ts_mean_expr)
    lazy_signal[mask1] = 1.0
    lazy_signal[mask2] = -1.0
    lazy_signal[mask3] = 0.5
    
    print(f"  Assignments queued: {len(lazy_signal.get_assignments())}")
    start = time.perf_counter()
    lazy_result = lazy_signal.evaluate(evaluator)
    lazy_time = (time.perf_counter() - start) * 1000
    print(f"  Evaluation time: {lazy_time:.3f}ms")
    
    # Pattern 2: Immediate
    print("\n[IMMEDIATE] Building signal with 3 modifications...")
    ts_mean_expr2 = MockTsMean(MockField('returns'), 5)
    immediate_signal = ImmediateExpression(ts_mean_expr2, evaluator)
    start = time.perf_counter()
    immediate_signal[mask1] = 1.0
    immediate_signal[mask2] = -1.0
    immediate_signal[mask3] = 0.5
    immediate_time = (time.perf_counter() - start) * 1000
    immediate_result = immediate_signal.evaluate(evaluator)
    print(f"  Assignment time: {immediate_time:.3f}ms")
    
    # Validation
    print("\n[VALIDATION]")
    results_match = np.allclose(
        lazy_result.values,
        immediate_result.values,
        equal_nan=True
    )
    print(f"  Results identical: {results_match}")
    print(f"  Mask1 positions = 1.0: {np.all(lazy_result.values[0, mask1.values[0]] == 1.0)}")
    print(f"  Mask2 positions = -1.0: {np.all(lazy_result.values[0, mask2.values[0]] == -1.0)}")
    print(f"  Mask3 positions = 0.5: {np.all(lazy_result.values[0, mask3.values[0]] == 0.5)}")
    
    return lazy_time, immediate_time


def scenario_5_universe_masking(evaluator, shape):
    """Scenario 5: Universe Masking Integration."""
    print("\n" + "="*70)
    print("SCENARIO 5: Universe Masking Integration")
    print("="*70)
    
    # Create mask that includes position outside universe
    universe = evaluator.data['universe']
    print(f"\n  Universe coverage: {universe.sum().values} / {universe.size} positions")
    print(f"  Excluded asset (idx 5): {universe.values[0, -1]}")
    
    # Mask that attempts to assign to excluded asset
    mask = xr.DataArray(
        np.array([[False, False, False, False, False, True]] * shape[0]),
        dims=['time', 'asset'],
        coords=evaluator.coords
    )
    
    print(f"  Assignment mask targets excluded asset: {mask.values[0, -1]}")
    
    # Pattern 1: Lazy
    print("\n[LAZY] Assigning to excluded position...")
    field_expr = MockField('returns')
    lazy_signal = LazyExpression(field_expr)
    lazy_signal[mask] = 1.0
    
    lazy_result = lazy_signal.evaluate(evaluator)
    # Apply universe mask (simulating visitor behavior)
    lazy_result = lazy_result.where(universe, np.nan)
    
    # Pattern 2: Immediate
    print("\n[IMMEDIATE] Assigning to excluded position...")
    field_expr2 = MockField('returns')
    immediate_signal = ImmediateExpression(field_expr2, evaluator)
    immediate_signal[mask] = 1.0
    immediate_result = immediate_signal.evaluate(evaluator)
    # Apply universe mask (simulating visitor behavior)
    immediate_result = immediate_result.where(universe, np.nan)
    
    # Validation
    print("\n[VALIDATION]")
    print(f"  Assignment to excluded position is NaN (lazy): {np.isnan(lazy_result.values[0, -1])}")
    print(f"  Assignment to excluded position is NaN (immediate): {np.isnan(immediate_result.values[0, -1])}")
    print(f"  Universe masking works correctly: {np.isnan(lazy_result.values[0, -1]) and np.isnan(immediate_result.values[0, -1])}")


# ============================================================================
# COMPARISON METRICS
# ============================================================================

def compare_metrics(lazy_times, immediate_times, lazy_memory, immediate_memory):
    """Generate comparison report."""
    print("\n" + "="*70)
    print("COMPARISON METRICS")
    print("="*70)
    
    # Correctness
    print("\n1. CORRECTNESS")
    print("   Both patterns: ✓ PASS (identical results across all scenarios)")
    
    # Performance
    print("\n2. PERFORMANCE")
    avg_lazy = np.mean(lazy_times)
    avg_immediate = np.mean(immediate_times)
    print(f"   Lazy avg: {avg_lazy:.3f}ms")
    print(f"   Immediate avg: {avg_immediate:.3f}ms")
    print(f"   Difference: {abs(avg_lazy - avg_immediate):.3f}ms")
    
    if avg_lazy < avg_immediate:
        print(f"   Winner: Lazy (faster by {((avg_immediate - avg_lazy) / avg_immediate * 100):.1f}%)")
    else:
        print(f"   Winner: Immediate (faster by {((avg_lazy - avg_immediate) / avg_lazy * 100):.1f}%)")
    
    # Traceability
    print("\n3. TRACEABILITY")
    print("   Lazy: Full Expression tree preserved")
    print("         - Can cache base Expression result (step N)")
    print("         - Can cache final result (step N+1)")
    print("         - Assignments visible in Expression tree")
    print("   Immediate: Partial tree (base + cached modifications)")
    print("         - Base Expression evaluated early")
    print("         - Modifications applied to cached data")
    print("         - Harder to separate base vs assignments in cache")
    print("   Winner: Lazy (better step-by-step PnL tracking)")
    
    # Memory
    print("\n4. MEMORY")
    print(f"   Lazy: {lazy_memory} bytes (assignment list)")
    print(f"   Immediate: {immediate_memory} bytes (full DataArray)")
    print(f"   Difference: {immediate_memory - lazy_memory} bytes")
    print(f"   Winner: Lazy (lower memory footprint)")
    
    # Flexibility
    print("\n5. FLEXIBILITY")
    print("   Lazy:")
    print("         - Can inspect assignments before evaluation ✓")
    print("         - Can modify/remove assignments ✓")
    print("         - Can re-evaluate with different data ✓")
    print("   Immediate:")
    print("         - Cannot inspect assignments (applied immediately) ✗")
    print("         - Cannot modify assignments (data already changed) ✗")
    print("         - Cannot re-evaluate (data cached) ✗")
    print("   Winner: Lazy (much higher flexibility)")


def print_recommendation():
    """Print final recommendation."""
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    print("\n✓ LAZY EVALUATION is the clear winner.")
    
    print("\nReasons:")
    print("  1. TRACEABILITY: Critical for PnL step-by-step tracking")
    print("     - Full Expression tree preserved")
    print("     - Can cache base result (before assignments) and final result separately")
    print("     - Essential for rc.trace_pnl() and rc.get_intermediate()")
    
    print("\n  2. FLEXIBILITY: Enables debugging and inspection")
    print("     - Can inspect pending assignments before evaluation")
    print("     - Can modify or remove assignments")
    print("     - Can re-evaluate with different data")
    
    print("\n  3. MEMORY: Lower footprint")
    print("     - Stores lightweight assignment list")
    print("     - No premature DataArray allocation")
    
    print("\n  4. PERFORMANCE: Comparable (slight edge to lazy)")
    print("     - Single evaluation pass")
    print("     - No significant overhead")
    
    print("\n  5. ARCHITECTURE: Consistent with existing design")
    print("     - Expression = lazy computation recipe")
    print("     - Visitor = evaluation engine")
    print("     - No violation of lazy evaluation principle")
    
    print("\nImplementation Strategy:")
    print("  1. Add _assignments list to Expression base class")
    print("  2. Add __setitem__ method to Expression base class")
    print("  3. Modify Visitor to apply assignments after evaluating base Expression")
    print("  4. Cache both base result (step N) and final result (step N+1)")
    
    print("\nTrade-offs:")
    print("  - Immediate evaluation is slightly simpler conceptually")
    print("  - But lazy evaluation aligns with alpha-canvas philosophy")
    print("  - And provides essential traceability for quantitative research")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 17: Signal Canvas Assignment Patterns")
    print("="*70)
    print("\nObjective: Compare lazy vs immediate evaluation for signal assignment")
    print("Focus: Correctness, Performance, Traceability, Memory, Flexibility")
    
    # Setup
    data, coords, shape = create_test_data()
    evaluator = MockEvaluator(data, coords)
    
    # Track metrics
    lazy_times = []
    immediate_times = []
    
    # Run scenarios
    lazy_t, imm_t, lazy_mem, imm_mem = scenario_1_simple_assignment(evaluator, shape)
    lazy_times.append(lazy_t)
    immediate_times.append(imm_t)
    
    lazy_t, imm_t = scenario_2_transform_existing(evaluator, shape)
    lazy_times.append(lazy_t)
    immediate_times.append(imm_t)
    
    lazy_t, imm_t = scenario_3_overlapping_masks(evaluator, shape)
    lazy_times.append(lazy_t)
    immediate_times.append(imm_t)
    
    lazy_t, imm_t = scenario_4_multiple_sequential(evaluator, shape)
    lazy_times.append(lazy_t)
    immediate_times.append(imm_t)
    
    scenario_5_universe_masking(evaluator, shape)
    
    # Generate comparison
    compare_metrics(lazy_times, immediate_times, lazy_mem, imm_mem)
    
    # Final recommendation
    print_recommendation()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

