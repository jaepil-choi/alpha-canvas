# Phase 6: Time-Series Operator - `ts_mean()` [REFACTORING]

## IMPORTANT: Design Pattern Refactoring

**Current Problem:** Visitor contains all computation logic, violating separation of concerns.

**Solution:** Operators own their `compute()` logic, Visitor only does traversal + caching.

## Goal

1. **Refactor** existing `ts_mean` to follow correct pattern (operator owns `compute()`)
2. Validate the operator implementation pattern before expanding to other operators

## Approach

Follow Experiment-Driven Development framework:

1. **Experiment** to validate the new pattern structure with `compute()` method
2. **Update documentation** (architecture.md, implementation.md) ✅ DONE  
3. **Write TDD tests** for both `compute()` and integrated Visitor behavior
4. **Refactor implementation** following the new pattern
5. **Update showcase** to demonstrate the correct pattern

---

## Step 1: Experiment - Validate New Pattern Structure

**File**: `experiments/exp_10_refactor_pattern.py`

**Objective**: Validate that separating `compute()` logic from Visitor works correctly

**What to validate**:
- Operator with `compute()` method can be tested independently
- Visitor can delegate to `compute()` successfully  
- No behavioral changes (results match original)
- Caching still works correctly
- Nested expressions work

**Success criteria**:
- `compute()` works when called directly (no Visitor)
- Visitor integration produces identical results to current implementation
- All edge cases still handled correctly
- Terminal output shows separation of concerns is clean

---

## Step 2: Update Findings

**File**: `experiments/FINDINGS.md`

Document the refactoring findings:
- Why separation of concerns matters
- Benefits of `compute()` pattern
- Testing improvements
- Code organization improvements

---

## Step 3: TDD - Write Tests for New Pattern

**File**: `tests/test_ops/test_timeseries.py` (UPDATE)

Add new tests for the refactored pattern:

```python
class TestTsMeanCompute:
    """Test TsMean.compute() method directly (no Visitor)."""
    
    def test_compute_basic(self):
        """Test compute() can be called directly."""
        data = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]],
            dims=['time', 'asset']
        )
        
        operator = TsMean(child=Field('dummy'), window=2)
        result = operator.compute(data)
        
        # Should work without any Visitor
        assert np.isnan(result.values[0, 0])
        assert result.values[1, 0] == 2.0  # mean([1, 3])
    
    def test_compute_is_pure_function(self):
        """Test compute() is a pure function (no side effects)."""
        data = xr.DataArray([[1, 2]], dims=['time', 'asset'])
        operator = TsMean(child=Field('dummy'), window=1)
        
        # Call twice
        result1 = operator.compute(data)
        result2 = operator.compute(data)
        
        # Results should be identical
        assert np.allclose(result1.values, result2.values)
        # Original data unchanged
        assert data.values[0, 0] == 1

class TestVisitorRefactored:
    """Test Visitor delegates to compute() correctly."""
    
    def test_visitor_delegates_to_compute(self):
        """Test visit_ts_mean() calls node.compute()."""
        # This ensures Visitor uses delegation pattern
```

**Keep existing tests** - they should all still pass after refactoring.

---

## Step 4: Refactor Implementation

### A. Update TsMean Expression

**File**: `src/alpha_canvas/ops/timeseries.py` (UPDATE)

```python
@dataclass
class TsMean(Expression):
    """Rolling time-series mean operator."""
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Visitor interface."""
        return visitor.visit_ts_mean(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """Core computation logic for rolling mean.
        
        Args:
            child_result: Input DataArray from child expression
        
        Returns:
            DataArray with rolling mean applied
        
        Note:
            This is a pure function with no side effects.
            Can be tested independently without Visitor.
        """
        return child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).mean()
```

### B. Refactor Visitor

**File**: `src/alpha_canvas/core/visitor.py` (UPDATE)

```python
def visit_ts_mean(self, node: TsMean) -> xr.DataArray:
    """Visit TsMean node - orchestrate traversal and caching.
    
    Visitor's role:
    1. Traverse tree (evaluate child)
    2. Delegate computation to operator
    3. Cache result
    
    Visitor does NOT contain computation logic.
    """
    # 1. Traversal: evaluate child expression
    child_result = node.child.accept(self)
    
    # 2. Delegation: operator does its own computation
    result = node.compute(child_result)
    
    # 3. State collection: cache result
    self._cache_result("TsMean", result)
    
    return result
```

---

## Step 5: Verify All Tests Pass

Run full test suite to ensure refactoring doesn't break anything:

```bash
poetry run pytest tests/ -v
```

Expected: All 65+ tests still pass

---

## Step 6: Update Showcase

**File**: `showcase/06_ts_mean_operator.py` (UPDATE)

Add section demonstrating the new pattern:

```python
print_section("NEW: Direct compute() Testing")

print("\n  Creating operator:")
operator = TsMean(child=Field('adj_close'), window=3)

print("\n  Testing compute() directly (no Visitor):")
sample_data = rc.db['close'].isel(time=slice(0, 5))
direct_result = operator.compute(sample_data)

print(f"  [OK] compute() works independently")
print(f"       Result shape: {direct_result.shape}")
print(f"       First value is NaN: {np.isnan(direct_result.values[0, 0])}")
```

---

## Success Criteria

Refactoring complete when:

- [x] Architecture docs updated with correct pattern
- [x] Implementation guide shows compute() pattern  
- [ ] Experiment validates new structure
- [ ] New tests for compute() method pass
- [ ] Refactored code passes all existing tests
- [ ] Showcase demonstrates the new pattern
- [ ] No behavioral changes (results identical)
- [ ] Code is cleaner and more testable

## Benefits Achieved

✅ **Separation of Concerns**: Operator logic separate from traversal  
✅ **Testability**: Can test `compute()` without Visitor  
✅ **Maintainability**: Each class has single responsibility  
✅ **Extensibility**: Adding operators doesn't bloat Visitor  

---

## Next Steps After Refactoring

Once refactoring is complete, this pattern can be used for:
- `ts_sum()` - rolling sum
- `ts_std()` - rolling standard deviation
- `ts_max()`, `ts_min()` - rolling extrema
- `ts_any()`, `ts_all()` - logical operators

All follow same structure:
1. Expression dataclass with `compute()` method
2. Visitor with 3-step pattern (traverse, delegate, cache)
3. Direct tests for `compute()` + integration tests


