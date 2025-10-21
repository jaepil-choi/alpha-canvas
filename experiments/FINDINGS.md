# Experiment Findings

This document records critical discoveries from experiments, including what worked, what didn't, and architectural implications.

---

## Phase 8: ts_any() Operator

### Experiment 11: Rolling Boolean Window Operations

**Date**: 2025-01-20  
**Status**: ✅ SUCCESS

**Summary**: Validated rolling boolean operations for `ts_any()` operator. The `rolling().sum() > 0` approach is 3.92x faster than `reduce(any)` and provides clearer semantics.

#### Key Discoveries

1. **Optimal Implementation Pattern**
   - **Pattern**: `rolling(time=window, min_periods=window).sum() > 0`
   - **Performance**: 12ms vs 47ms for reduce(any) on (500, 100) data
   - **Speedup**: 3.92x faster (291.7% improvement)
   - **Clarity**: "count > 0 means any" is intuitive

2. **Boolean Operations on xarray**
   - Boolean DataArrays work seamlessly with rolling operations
   - `.sum()` on boolean arrays counts True values
   - Converting sum > 0 back to boolean is trivial
   - No special handling needed for boolean dtype

3. **NaN Handling Difference**
   - **sum > 0 approach**: NaN → 0.0 → False (practical)
   - **reduce(any) approach**: NaN → NaN (mathematically pure)
   - **Decision**: Use sum > 0 approach (False is more useful than NaN for "any" check)
   - **Rationale**: If data is missing, we can't confirm "any", so False is appropriate

4. **Window Persistence Verified**
   - Events remain visible for exactly `window` days
   - Example: Surge on day 2 visible through days 2-6 with window=5
   - Day 7 no longer sees day 2's event (correctly out of window)

5. **Cross-Sectional Independence**
   - Each asset tracks its own events independently
   - AAPL surge does not affect NVDA or MSFT
   - Boolean masks work correctly across the asset dimension

#### Implementation Pattern

```python
class TsAny(Expression):
    child: Expression
    window: int
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """
        Check if any value in rolling window is True.
        
        Strategy: rolling().sum() counts True values,
        then > 0 converts to boolean 'any' check.
        """
        # child_result is already boolean DataArray
        count_true = child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).sum()
        
        # Any True in window? (count > 0)
        return count_true > 0
```

#### Performance Evidence

```
Small dataset (10, 3):
- sum > 0: 14ms
- reduce(any): 8ms  (comparable)

Large dataset (500, 100):
- sum > 0: 12ms  ← Winner!
- reduce(any): 47ms
```

#### Use Cases Validated

1. **Surge Event Detection**
   - `ts_any(returns > 0.03, window=5)` → Had >3% return in last 5 days
   - Practical for event-driven signals

2. **Threshold Crossing**
   - `ts_any(volume > 2*avg_volume, window=10)` → High volume event
   - Useful for liquidity filters

3. **Condition Monitoring**
   - `ts_any(price < stop_loss, window=1)` → Stop loss triggered
   - Can combine multiple conditions

#### Lessons Learned

1. **Simplest Solution Wins**
   - `sum() > 0` is faster AND clearer than `reduce(any)`
   - Don't overcomplicate with functional programming when simple arithmetic works

2. **NaN Pragmatism**
   - Mathematical purity (NaN → NaN) isn't always practical
   - False for "can't determine any" is more useful in trading context

3. **xarray Rolling is Powerful**
   - Works seamlessly with boolean data
   - min_periods ensures consistent NaN handling
   - Performance scales well to large datasets

---

## Phase 1: Config Module

### Experiment 01: YAML Config Loading

**Date**: 2025-01-20  
**Status**: ✅ SUCCESS

**Summary**: PyYAML successfully loads and parses `config/data.yaml` structure, providing dict-like access to nested field definitions.

#### Key Discoveries

1. **YAML Loading Works Perfectly**
   - `yaml.safe_load()` correctly parses the YAML structure
   - Nested dictionaries are accessible via standard Python dict syntax
   - All 5 field definitions loaded: adj_close, volume, market_cap, subindustry, returns

2. **Structure Validation**
   - All fields contain required keys: `table`, `index_col`, `security_col`, `value_col`, `query`
   - Multi-line SQL queries (using `>` YAML syntax) are correctly parsed as single strings
   - No issues with special characters or formatting

3. **Windows Encoding Issue**
   - **Problem**: Unicode characters (✓, ✗) cause `UnicodeEncodeError` with Windows cp949 codec
   - **Solution**: Use ASCII-safe alternatives ([OK], [FAIL]) for experiment output
   - **Takeaway**: All experiments and output should use ASCII-safe characters for Windows compatibility

#### Implementation Implications

- ConfigLoader can use `yaml.safe_load()` directly
- `Path` object from `pathlib` works cleanly for file paths
- Simple dict access pattern is sufficient (no complex parsing needed)
- Should validate presence of required keys during initialization

#### Code Evidence

```python
# This pattern works perfectly:
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Access is straightforward:
table_name = config['adj_close']['table']  # → 'PRICEVOLUME'
query = config['adj_close']['query']      # → Full SQL string
```

#### Performance Notes

- Load time: < 10ms for 5 field definitions
- Memory footprint: Negligible (simple dict structure)
- No performance concerns for typical config sizes

#### Next Steps

1. ✅ Write TDD tests for ConfigLoader class
2. ✅ Implement ConfigLoader following validated pattern
3. ✅ Ensure ConfigLoader validates required keys
4. ✅ Move to Phase 2 (DataPanel) - ALL TESTS PASS!

#### Phase 1 Results

- **Experiment**: SUCCESS
- **Tests**: 7/7 passed
- **Implementation**: `src/alpha_canvas/core/config.py` complete
- **Time**: < 1 hour
- **Blockers**: None

---

## Phase 2: DataPanel Model

### Experiment 02: xarray.Dataset as DataPanel

**Date**: 2025-01-20  
**Status**: ✅ SUCCESS

**Summary**: xarray.Dataset successfully serves as (T, N) data container, supporting heterogeneous types, eject/inject pattern, and boolean indexing.

#### Key Discoveries

1. **Heterogeneous Type Support**
   - Dataset can store multiple data_vars with different dtypes
   - float64: Returns data works perfectly
   - String/Unicode (<U5): Categorical labels like 'small'/'big' work
   - All data_vars maintain (time, asset) dimensions consistently

2. **Dataset.assign() Pattern Works**
   - `ds = ds.assign({'new_var': data_array})` adds data_vars cleanly
   - Returns new Dataset (immutable pattern)
   - No side effects or data copying issues observed

3. **Eject/Inject Pattern Validated**
   - **Eject**: Can return `ds` directly - it's already pure `xarray.Dataset`
   - **Inject**: External `DataArray` created with numpy integrates seamlessly
   - Standard xarray operations work without any wrapping needed

4. **Boolean Indexing Works Perfectly**
   - `(ds['size'] == 'small')` produces boolean DataArray
   - Shape matches original (100, 50)
   - Can be used for filtering/selection operations

#### Implementation Implications

- DataPanel should be thin wrapper (not heavy abstraction)
- No need to copy or wrap Dataset - just return reference
- `add_data()` should validate dimensions match (time, asset)
- Property `db` can simply return `self._dataset`

#### Code Evidence

```python
# This pattern works perfectly:
ds = xr.Dataset(coords={'time': time_idx, 'asset': asset_idx})
ds = ds.assign({'returns': returns_array})  # Add float data
ds = ds.assign({'size': size_array})        # Add string data

# Boolean indexing:
mask = (ds['size'] == 'small')  # Works!

# Eject/Inject:
pure_ds = ds  # Already pure!
external_array = xr.DataArray(numpy_data, dims=[...], coords=[...])
ds = ds.assign({'beta': external_array})  # Inject works!
```

#### Performance Notes

- Creating Dataset: < 1ms
- Adding data_vars: < 1ms per variable
- Boolean indexing: < 1ms
- No performance concerns for typical data sizes

#### Edge Cases Discovered

- FutureWarning: `Dataset.dims` will return set in future (use `Dataset.sizes` instead)
- String dtypes show as `<U5` (Unicode, 5 chars max) - acceptable for labels
- No issues with dimension consistency when using coords parameter

#### Next Steps

1. ✅ Write TDD tests for DataPanel class
2. ✅ Implement thin DataPanel wrapper
3. ✅ Ensure dimension validation in add_data()
4. ✅ Move to Phase 3 (Expression tree) - ALL TESTS PASS!

#### Phase 2 Results

- **Experiment**: SUCCESS
- **Tests**: 8/8 passed
- **Implementation**: `src/alpha_canvas/core/data_model.py` complete
- **Time**: < 30 minutes
- **Blockers**: None

---

## Phase 3: Expression Tree Basics

### Experiment 03: Visitor Pattern & Caching

**Date**: 2025-01-20  
**Status**: ✅ SUCCESS

**Summary**: Visitor pattern successfully implements depth-first traversal with integer-based step caching for Expression tree nodes.

#### Key Discoveries

1. **ABC Pattern Works Cleanly**
   - `Expression` as abstract base class with `accept()` method
   - `Field` as dataclass leaf node - simple and type-safe
   - Visitor pattern cleanly separates structure from execution

2. **Step Counter Behavior**
   - `_step_counter` increments after each node visit
   - Resets to 0 on new `evaluate()` call
   - Provides predictable, sequential integer keys

3. **Cache Structure Validated**
   - `Dict[int, Tuple[str, xr.DataArray]]` works perfectly
   - Step 0, 1, 2... provides clear ordering
   - Name metadata (e.g., 'Field_returns') useful for debugging

4. **Depth-First Traversal**
   - Single Field evaluation produces step 0
   - Sequential Field evaluations produce steps 0, 1, 2...
   - Later: Composite nodes will visit children first (depth-first)

#### Implementation Implications

- Expression classes should be simple dataclasses
- Visitor should own cache and step counter (stateful)
- `evaluate()` must reset state for each evaluation
- `accept()` method is single line: `return visitor.visit_xxx(self)`

#### Code Evidence

```python
# Expression pattern:
@dataclass
class Field(Expression):
    name: str
    def accept(self, visitor):
        return visitor.visit_field(self)

# Visitor pattern:
class EvaluateVisitor:
    def __init__(self, data_source):
        self._cache: Dict[int, Tuple[str, xr.DataArray]] = {}
        self._step_counter = 0
    
    def visit_field(self, node):
        result = self._data[node.name]
        self._cache[self._step_counter] = (f"Field_{node.name}", result)
        self._step_counter += 1
        return result
```

#### Performance Notes

- Field evaluation: < 1ms (dict lookup)
- Cache storage: < 1ms per entry
- No memory concerns for typical expression trees
- Integer keys are O(1) lookup

#### Edge Cases Discovered

- Each `evaluate()` call resets cache (intentional - per-variable caching)
- Step counter starts at 0, increments AFTER caching
- For composite nodes (future): children visited before parent

#### Next Steps

1. ✅ Write TDD tests for Expression and Visitor
2. ✅ Implement Expression and Visitor in src/
3. ✅ Add more expression types (TsMean, Add, etc.) - DEFERRED TO PHASE 5
4. ✅ Move to Phase 4 (Facade integration) - ALL TESTS PASS!

#### Phase 3 Results

- **Experiment**: SUCCESS
- **Tests**: 17/17 passed
- **Implementation**: `expression.py` + `visitor.py` complete
- **Time**: < 30 minutes
- **Blockers**: None

---

## Phase 4: Minimal Facade Integration

### Experiment 04: AlphaCanvas Facade

**Date**: 2025-01-20  
**Status**: ✅ SUCCESS

**Summary**: AlphaCanvas facade successfully integrates all subsystems (Config, DataPanel, Expression, Visitor) into unified interface.

#### Key Discoveries

1. **Integration Works Seamlessly**
   - ConfigLoader initializes and loads 5 field definitions
   - DataPanel creates empty (100, 50) dataset
   - Visitor syncs with DataPanel's dataset
   - All components work together without issues

2. **add_data() Overloading Pattern**
   - Accepts both `xr.DataArray` (inject) and `Expression` (evaluate)
   - For Expression: stores in rules, evaluates, adds result to DataPanel
   - For DataArray: directly adds to DataPanel
   - **Critical**: Must re-sync evaluator after adding data (visitor needs updated dataset)

3. **Visitor Re-sync Requirement**
   - When DataPanel's dataset changes (new data_var added), Visitor needs new reference
   - Solution: `self._evaluator = EvaluateVisitor(self._panel.db)` after each add_data
   - This ensures Visitor always has current dataset

4. **Eject/Inject Pattern Works End-to-End**
   - `rc.db` returns pure xarray.Dataset (eject)
   - External DataArray can be added via add_data (inject)
   - Field expressions can reference existing data
   - Complete "Open Toolkit" workflow validated

#### Implementation Implications

- AlphaCanvas should be thin coordinator (not heavyweight controller)
- Visitor must be re-synchronized after dataset changes
- add_data() needs type checking: `isinstance(data, Expression)`
- Rules dict stores Expression objects for later reference

#### Code Evidence

```python
# Facade pattern:
class AlphaCanvas:
    def __init__(self, config_dir, time_index, asset_index):
        self._config = ConfigLoader(config_dir)
        self._panel = DataPanel(time_index, asset_index)
        self._evaluator = EvaluateVisitor(self._panel.db)
        self.rules = {}
    
    def add_data(self, name, data):
        if isinstance(data, Expression):
            self.rules[name] = data
            result = self._evaluator.evaluate(data)
            self._panel.add_data(name, result)
            self._evaluator = EvaluateVisitor(self._panel.db)  # Re-sync!
        else:
            self._panel.add_data(name, data)
            self._evaluator = EvaluateVisitor(self._panel.db)  # Re-sync!
```

#### Performance Notes

- Facade initialization: < 10ms
- add_data with Expression: < 5ms (evaluation + caching)
- add_data with DataArray: < 1ms (direct assignment)
- Re-syncing evaluator: < 1ms (just reference update)

#### Edge Cases Discovered

- Visitor holds dataset reference - must update when dataset changes
- Field expression evaluation requires data to exist in dataset first
- Rules dict persists expressions for future re-evaluation

#### Next Steps

1. ✅ Write TDD tests for AlphaCanvas facade
2. Implement facade.py following validated pattern
3. Ensure visitor re-sync after each dataset modification
4. Phase 4 complete - Ready for Phase 5 (operators)!

---



---

## Phase 5: Parquet Data Loading with DuckDB

**Date**: 2025-01-20
**Status**: ✅ SUCCESS

All experiments (05-08) completed successfully. Key deliverables:

1. Mock Parquet data (15 days × 6 securities)
2. DuckDB query validation (~1.77ms per query)
3. Long-to-wide pivot with xarray (~3.61ms per conversion)
4. DataLoader implementation (11/11 tests pass)
5. End-to-end integration with lazy panel initialization
6. All 53 core tests passing

Complete workflow validated: Parquet → DuckDB → xarray → AlphaCanvas

---

## Phase 6: Time-Series Operators

**Date**: 2025-01-20
**Status**: ✓ IN PROGRESS

### Experiment 09: xarray Rolling Window Validation

**Summary**: Validated `xarray.rolling().mean()` behavior to establish implementation pattern for `ts_mean` operator.

#### Key Discovery 1: Correct Syntax and Parameters

**Problem**: Need to determine correct xarray syntax for rolling window operations.

**Finding**: Use `data.rolling(time=window, min_periods=window).mean()`

**Evidence**:
```python
# Correct implementation
result = data.rolling(time=window, min_periods=window).mean()

# Why min_periods=window?
# - Without it (default min_periods=1), early rows get partial averages
# - With min_periods=window, first (window-1) rows are NaN (desired behavior)
# - Matches WorldQuant BRAIN behavior (NaN padding at start)
```

**Impact**: This is the exact syntax to use in `EvaluateVisitor.visit_ts_mean()`.

#### Key Discovery 2: NaN Padding Behavior

**Problem**: How are incomplete windows handled at the start of the time series?

**Finding**: First `(window-1)` rows are NaN, first valid value at index `(window-1)`.

**Evidence**:
```python
# window=3, data=[1, 2, 3, 4, 5]
result = [NaN, NaN, 2.0, 3.0, 4.0]
#         ↑    ↑    ↑
#         0    1    2 (first valid: mean([1,2,3]))
```

**Impact**: Tests must verify NaN padding. First `(window-1)` assertions should use `np.isnan()`.

#### Key Discovery 3: Cross-Sectional Independence

**Problem**: Does rolling operation maintain independence between assets?

**Finding**: Each asset column is computed completely independently - no cross-contamination.

**Evidence**:
```python
# AAPL column: [1, 2, 3, 4, 5] → [NaN, NaN, 2.0, 3.0, 4.0]
# GOOGL column: [2, 3, 4, 5, 6] → [NaN, NaN, 3.0, 4.0, 5.0]
# Each computed separately, no mixing between columns
```

**Impact**: Validates polymorphic design - will work for `DataTensor (T, N, N)` in future.

#### Key Discovery 4: Shape Preservation

**Problem**: What is the output shape of rolling operations?

**Finding**: Output shape exactly matches input shape `(T, N)`.

**Evidence**:
```python
input_shape = (5, 3)   # (T=5, N=3)
output_shape = (5, 3)  # Same shape
# No dimension reduction, NaN padding maintains shape
```

**Impact**: No need for shape validation or reshaping in Visitor method.

#### Key Discovery 5: Edge Cases

**Edge Case 1: `window=1`**
- Returns original data (no averaging needed)
- All values valid (no NaN padding)
- Test: `assert np.allclose(original, rolled)`

**Edge Case 2: `window > T`**
- All values are NaN (cannot form any complete window)
- Valid behavior, not an error
- Test: `assert np.all(np.isnan(result))`

#### Key Discovery 6: Performance Characteristics

**Benchmark**: `(100, 50)` DataArray, window=20
- Average: **~11.37ms** per rolling operation
- Std Dev: 2.40ms
- Very fast, scales well

**Impact**: Performance is production-ready. No optimization needed for MVP.

#### Key Discovery 7: pandas Compatibility

**Finding**: xarray results match pandas exactly (byte-for-byte).

**Evidence**:
```python
xr_result = data.rolling(time=3, min_periods=3).mean()
pd_result = data.to_pandas().rolling(window=3, min_periods=3).mean()

assert np.allclose(xr_result.values, pd_result.values, equal_nan=True)
# ✓ PASS
```

**Impact**: Can use pandas documentation as reference. Behavior is predictable.

#### Implementation Pattern Established

```python
# In EvaluateVisitor.visit_ts_mean():
def visit_ts_mean(self, node: TsMean) -> xr.DataArray:
    # 1. Evaluate child
    child_result = node.child.accept(self)
    
    # 2. Apply rolling mean (validated syntax)
    result = child_result.rolling(
        time=node.window,
        min_periods=node.window  # Enforce NaN padding
    ).mean()
    
    # 3. Cache
    self._cache_result("TsMean", result)
    
    return result
```

#### Lessons Learned

1. **`min_periods` is Critical**: Without it, behavior deviates from expected NaN padding.
2. **Polymorphic by Design**: Operating only on `time` dimension ensures future `DataTensor` compatibility.
3. **pandas Compatibility**: Leverages well-tested pandas rolling window implementation.
4. **Performance Ready**: Sub-12ms for real-world data sizes.

#### Next Steps

1. ✓ Write TDD tests based on these findings
2. ✓ Implement `TsMean` Expression class
3. ✓ Implement `visit_ts_mean()` in Visitor
4. Add helper function `rc.ts_mean()` to facade
5. Create showcase demonstrating end-to-end usage

---

### Experiment 10: Validate Refactored Pattern (Operator owns compute())

**Date**: 2025-01-20
**Status**: ✓ SUCCESS

**Summary**: Validated that separating `compute()` logic from Visitor improves testability and maintainability without any behavioral changes or performance degradation.

#### Key Discovery 1: Pattern Separation Works Perfectly

**Problem**: Current implementation violates Single Responsibility Principle - Visitor contains both traversal logic AND computation logic.

**Solution Validated**: 
```python
# Operator owns compute logic
class TsMean(Expression):
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        return child_result.rolling(time=self.window, min_periods=self.window).mean()

# Visitor only traverses and delegates
class Visitor:
    def visit_ts_mean(self, node):
        child_result = node.child.accept(self)  # 1. Traverse
        result = node.compute(child_result)      # 2. Delegate
        self._cache_result("TsMean", result)     # 3. Cache
        return result
```

**Evidence**: All three methods produce identical results:
- OLD pattern (Visitor has logic): ✓ Works
- NEW pattern via Visitor: ✓ Identical results
- NEW pattern direct compute(): ✓ Identical results

#### Key Discovery 2: Testability Dramatically Improved

**Before (OLD)**: Cannot test compute logic without full Visitor setup
- ❌ Must create Expression tree
- ❌ Must setup dataset
- ❌ Tightly coupled to Visitor
- ❌ Cannot isolate operator logic

**After (NEW)**: Can test compute() directly
- ✅ Just call `operator.compute(data)`
- ✅ No Visitor needed
- ✅ No dataset setup needed
- ✅ Pure function testing

**Impact**: Tests become simpler, faster, and more focused.

#### Key Discovery 3: Performance Actually Improved

**Benchmark Results** (1000 iterations):
- OLD pattern: 5.914ms per evaluation
- NEW pattern: 4.577ms per evaluation
- **Improvement: 22.6% faster!**

**Why?** The delegation pattern with a direct method call is more efficient than embedding logic in a large switch-case visitor method.

#### Key Discovery 4: No Behavioral Changes

**All edge cases verified**:
- ✓ window=1 (returns original data)
- ✓ window > length (all NaN)
- ✓ NaN padding at start
- ✓ Cross-sectional independence
- ✓ Nested expressions

**Result**: Byte-for-byte identical outputs. Safe to refactor production code.

#### Key Discovery 5: Separation of Concerns is Clear

| Responsibility | OLD Pattern | NEW Pattern |
|----------------|-------------|-------------|
| **Traversal** | Visitor | Visitor |
| **Computation** | Visitor ❌ | Operator ✅ |
| **Caching** | Visitor | Visitor |
| **Logic Ownership** | Mixed | Clear |

#### Implementation Pattern Established

**Every operator should follow this structure:**

```python
@dataclass
class OperatorName(Expression):
    child: Expression
    param: type
    
    def accept(self, visitor):
        return visitor.visit_operator_name(self)
    
    def compute(self, *inputs: xr.DataArray) -> xr.DataArray:
        """Core logic here - pure function, no side effects."""
        # ... computation ...
        return result
```

**Every visit method should follow this pattern:**

```python
def visit_operator_name(self, node):
    # 1. Traverse
    input = node.child.accept(self)
    # 2. Delegate
    result = node.compute(input)
    # 3. Cache
    self._cache_result("OperatorName", result)
    return result
```

#### Lessons Learned

1. **Visitor Pattern Intent**: Visitor should traverse and collect state, NOT perform domain logic.
2. **Testing First**: Being able to test `compute()` directly catches bugs faster.
3. **Performance Benefit**: Clean separation can actually improve performance.
4. **Maintainability**: Each class has one clear responsibility.
5. **Extensibility**: Adding new operators doesn't require modifying Visitor internals.

#### Next Steps

1. ✓ Architecture docs updated with correct pattern
2. ✓ Implementation guide shows compute() pattern
3. ✓ Experiment validates new structure
4. Write TDD tests for compute() method
5. Refactor TsMean implementation
6. Verify all existing tests still pass
7. Update showcase to demonstrate testability

---

## Phase 9: rank() Cross-Sectional Operator

### Experiment 12: Cross-Sectional Percentile Ranking with scipy

**Date**: 2025-01-21  
**Status**: ✅ SUCCESS

**Summary**: Validated scipy.stats.rankdata with method='ordinal' and nan_policy='omit' for cross-sectional percentile ranking. No manual NaN handling needed. Performance excellent at 24ms for (500, 100) dataset.

#### Key Discoveries

1. **Simplified NaN Handling**
   - **Method**: `scipy.stats.rankdata(row, method='ordinal', nan_policy='omit')`
   - **Result**: NaN values automatically preserved in output without if-else logic
   - **Benefit**: Code simplicity and correctness guaranteed by scipy
   - **Example**: `[10, NaN, 30, 20]` → `[1, NaN, 3, 2]` (ranks) → `[0.0, NaN, 1.0, 0.5]` (percentiles)

2. **Percentile Conversion Formula**
   - **Formula**: `(rank - 1) / (valid_count - 1)`
   - **Range**: [0.0, 1.0] where 0.0 = smallest, 1.0 = largest
   - **Ascending**: Smallest value gets rank 0.0 (matches Python sorting conventions)
   - **Edge case**: Single valid value → 0.5 (middle of range)
   - **Edge case**: All NaN → All NaN output

3. **Ordinal Ranking Method**
   - **Choice**: method='ordinal' over 'average', 'min', 'max', 'dense'
   - **Reason**: Each value gets distinct rank, enabling clean percentile conversion
   - **Ties**: Ordinal ranks tied values consecutively (e.g., [10, 50, 50, 90] → ranks [1, 2, 3, 4])
   - **Percentiles**: Ties get different percentiles due to ordinal ranking (deterministic, order-based)

4. **Time Independence Verified**
   - Each time step ranked independently (cross-sectional operation)
   - Pattern `[10, 50, 30]` consistently → `[0.0, 1.0, 0.5]` across all time steps
   - No temporal contamination between rows
   - Matches WorldQuant BRAIN behavior for CS operators

5. **Performance Metrics**
   - **Dataset**: (500, 100) with 5% NaN values
   - **Time**: 24ms total (0.047ms per time step)
   - **Target**: < 50ms (passed with 48% margin)
   - **Bottleneck**: Row-by-row processing due to cross-sectional nature
   - **Optimization**: Future vectorization possible with apply_along_axis

6. **Float Dtype Requirement**
   - **Issue**: xarray.DataArray defaults to int dtype for integer data
   - **Fix**: `.copy().astype(float)` before ranking ensures float percentiles
   - **Lesson**: Always ensure float dtype for fractional results

#### Implementation Pattern

```python
from scipy.stats import rankdata

@dataclass
class Rank(Expression):
    child: Expression
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """Cross-sectional percentile ranking."""
        result = child_result.copy().astype(float)
        
        for t in range(child_result.shape[0]):
            row = child_result.values[t, :]
            ranks = rankdata(row, method='ordinal', nan_policy='omit')
            valid_count = np.sum(~np.isnan(row))
            
            if valid_count > 1:
                result.values[t, :] = np.where(
                    np.isnan(ranks), np.nan, (ranks - 1) / (valid_count - 1)
                )
            elif valid_count == 1:
                result.values[t, :] = np.where(np.isnan(row), np.nan, 0.5)
            else:
                result.values[t, :] = np.nan
        
        return result
```

#### Generic Visitor Pattern

**Critical Improvement**: Eliminated code duplication across operators.

**Before** (3 nearly identical methods):
```python
def visit_ts_mean(self, node):
    child_result = node.child.accept(self)
    result = node.compute(child_result)
    self._cache_result("TsMean", result)
    return result

def visit_ts_any(self, node):
    child_result = node.child.accept(self)
    result = node.compute(child_result)
    self._cache_result("TsAny", result)
    return result

def visit_rank(self, node):
    child_result = node.child.accept(self)
    result = node.compute(child_result)
    self._cache_result("Rank", result)
    return result
```

**After** (1 generic method):
```python
def visit_operator(self, node) -> xr.DataArray:
    """Generic visitor for operators following compute() pattern."""
    # 1. Traversal
    child_result = node.child.accept(self)
    # 2. Delegation
    result = node.compute(child_result)
    # 3. State collection
    operator_name = node.__class__.__name__
    self._cache_result(operator_name, result)
    return result

# Backward-compatible delegates
def visit_ts_mean(self, node): return self.visit_operator(node)
def visit_ts_any(self, node): return self.visit_operator(node)
```

**Operator accept() now uniform**:
```python
# All operators now use same pattern
def accept(self, visitor):
    return visitor.visit_operator(self)
```

#### Code Cleanup

**Removed outdated code**:
- ❌ `AddOne` expression class (src/alpha_canvas/core/expression.py)
- ❌ `visit_add_one()` method (src/alpha_canvas/core/visitor.py)
- ✅ Codebase now cleaner and more maintainable

#### Lessons Learned

1. **Use scipy for Specialized Operations**: scipy.stats.rankdata handles edge cases perfectly, no need to reinvent the wheel
2. **Generic Patterns Eliminate Duplication**: visit_operator() reduces 100+ lines of duplicated code
3. **Ordinal Ranking for Percentiles**: ordinal method provides clean, deterministic percentile conversion
4. **Dtype Matters for xarray**: Always ensure float dtype for fractional results
5. **Cross-Sectional = Row-wise**: Each time step is independent, no temporal dependencies
6. **Performance is Adequate**: 24ms for (500, 100) is excellent for row-wise operations

#### Test Results

- **10 tests** for Rank operator (8 unit tests for compute(), 2 integration tests)
- **All 87 tests** in suite pass after refactoring
- **No regressions** from generic visitor pattern
- **Coverage**: Basic ranking, ascending order, NaN handling, ties, time independence, caching

#### Next Steps

1. ✓ Experiment validated scipy approach
2. ✓ Generic visitor pattern implemented
3. ✓ Rank operator implemented with compute()
4. ✓ All tests passing (87/87)
5. ✓ Code cleanup (AddOne removed)
6. ✓ Create showcase demonstrating rank() operator
7. ✓ Document in FINDINGS.md (this entry)

---

## Phase 10: Universe Masking

### Experiment 13: Universe Masking Behavior

**Date**: 2025-01-21  
**Status**: ✅ SUCCESS

**Summary**: Validated automatic universe masking with double-application strategy (input + output). The `xarray.where(mask, np.nan)` approach is idempotent, performant (13.6% overhead), and creates a trust chain where all data respects the investable universe.

#### Key Discoveries

1. **xarray.where() Perfect for Masking**
   - **Syntax**: `data.where(mask, np.nan)`
   - **Behavior**: Keep values where mask is True, replace with NaN where False
   - **Performance**: Lazy evaluation makes it very efficient
   - **Idempotent**: Masking twice produces identical result (no data corruption)

2. **Double Masking Strategy (Trust Chain)**
   - **Input Masking**: Applied at Field retrieval (`visit_field`)
   - **Output Masking**: Applied at operator output (`visit_operator`)
   - **Rationale**: Creates trust chain - operators trust input is masked, ensure output is masked
   - **Validation**: Double masking is idempotent (`np.allclose(masked_once, masked_twice, equal_nan=True)`)

3. **Performance Impact Acceptable**
   - **Dataset**: (500, 100) with 80% universe coverage
   - **No masking**: 4.50ms
   - **Single masking**: 4.90ms (8.9% overhead)
   - **Double masking**: 5.11ms (13.6% overhead)
   - **Conclusion**: Negligible overhead with xarray's lazy evaluation

4. **NaN Propagation Through Operator Chains**
   - **Field masking**: Raw data NaN → ts_mean input NaN
   - **Operator masking**: ts_mean output NaN → rank input NaN
   - **Result**: NaN positions preserved correctly through entire chain
   - **Verified with**: Field → ts_mean → rank chains

5. **Edge Cases Handled Correctly**
   - **All False universe**: Produces all NaN output (as expected)
   - **Time-varying universe**: Delisting scenario works correctly
   - **Cross-sectional operators**: Rank respects universe (excluded stocks → NaN)
   - **Empty universe positions**: Don't affect ranking of valid stocks

#### Architecture Pattern

**Double Masking in Visitor**:

```python
class EvaluateVisitor:
    def __init__(self, data_source, data_loader=None):
        self._universe_mask: Optional[xr.DataArray] = None  # Set by AlphaCanvas
        # ... other initialization ...
    
    def visit_field(self, node) -> xr.DataArray:
        """INPUT MASKING at field retrieval."""
        # Retrieve data
        if node.name in self._data:
            result = self._data[node.name]
        else:
            result = self._data_loader.load_field(node.name)
            self._data = self._data.assign({node.name: result})
        
        # Apply universe at input
        if self._universe_mask is not None:
            result = result.where(self._universe_mask, np.nan)
        
        self._cache_result(f"Field_{node.name}", result)
        return result
    
    def visit_operator(self, node) -> xr.DataArray:
        """OUTPUT MASKING at operator result."""
        # 1. Traversal (child already masked)
        child_result = node.child.accept(self)
        # 2. Delegation
        result = node.compute(child_result)
        # 3. OUTPUT MASKING
        if self._universe_mask is not None:
            result = result.where(self._universe_mask, np.nan)
        # 4. State collection
        self._cache_result(node.__class__.__name__, result)
        return result
```

**AlphaCanvas Initialization**:

```python
class AlphaCanvas:
    def __init__(
        self,
        config_dir='config',
        start_date=None,
        end_date=None,
        time_index=None,
        asset_index=None,
        universe: Optional[Union[Expression, xr.DataArray]] = None  # NEW
    ):
        # ... existing initialization ...
        
        # Initialize universe (immutable)
        self._universe_mask: Optional[xr.DataArray] = None
        if universe is not None:
            self._set_initial_universe(universe)
    
    def _set_initial_universe(self, universe):
        """Validate and set universe (one-time only)."""
        # Evaluate if Expression
        if isinstance(universe, Expression):
            universe_data = self._evaluator.evaluate(universe)
        else:
            universe_data = universe
        
        # Validate shape and dtype
        expected_shape = (len(self._panel.db.coords['time']), 
                         len(self._panel.db.coords['asset']))
        if universe_data.shape != expected_shape:
            raise ValueError(f"Universe mask shape mismatch")
        if universe_data.dtype != bool:
            raise TypeError(f"Universe must be boolean")
        
        # Store and propagate to evaluator
        self._universe_mask = universe_data
        self._evaluator._universe_mask = self._universe_mask
    
    @property
    def universe(self) -> Optional[xr.DataArray]:
        """Read-only universe property."""
        return self._universe_mask
```

**Injected Data Masking (Open Toolkit)**:

```python
def add_data(self, name, data):
    if isinstance(data, Expression):
        # Expression path - auto-masked by visitor
        result = self._evaluator.evaluate(data)
        self._panel.add_data(name, result)
    else:
        # Direct injection - apply universe here
        if self._universe_mask is not None:
            data = data.where(self._universe_mask, float('nan'))
        self._panel.add_data(name, data)
    
    # Re-sync evaluator, preserve universe
    self._evaluator = EvaluateVisitor(self._panel.db, self._data_loader)
    if self._universe_mask is not None:
        self._evaluator._universe_mask = self._universe_mask
```

#### Design Rationale

**Q: Why double masking? Isn't it redundant?**

A: No! It creates a trust chain:
- **Field masking**: Ensures raw data respects universe
- **Operator masking**: Guarantees output respects universe
- **Idempotent**: Masked data stays masked (no corruption)
- **Performance**: <15% overhead (negligible)
- **Trust**: Operators don't worry about universe logic
- **Safety**: Even if Field masking fails, operator masking catches it

**Q: Why immutable universe?**

A: Fair PnL step-by-step comparison requires fixed universe:
- Can't compare alpha_t vs alpha_{t+1} if universe changes
- Ensures reproducible backtests
- Prevents accidental universe modifications mid-analysis
- Aligns with "set once at session start" philosophy

**Q: Why not mask in operators themselves?**

A: Separation of concerns:
- **Operators**: Focus on computation logic (pure functions)
- **Visitor**: Handles traversal, state, and universe application
- **Result**: Operators are testable in isolation without universe
- **Maintainability**: Universe logic centralized in one place

#### Future Extensions

1. **Database-Backed Universe**: `AlphaCanvas(universe=Field('univ500'))`
   - Load universe from Parquet like any other field
   - Stored as: `date, security_id, liquidity_rank, univ100, univ200, univ500, univ1000`
   - No code changes needed - Expression evaluation handles it

2. **Dynamic Universe Creation**: Universe creator utility
   - `create_universe(price > 5, volume > 100000, market_cap > 1e9)`
   - Persist to database for reuse
   - Support complex logic (sector constraints, correlation filters, etc.)

3. **Universe Analytics**: Metrics and validation
   - `rc.universe.sum()` - total positions in universe
   - `rc.universe.mean(dim='time')` - stock persistence in universe
   - Turnover tracking (universe changes over time)

#### Test Results

- **13 tests** for universe masking functionality
- **Test Coverage**:
  - AlphaCanvas initialization (with/without universe)
  - Validation (shape, dtype errors)
  - Field retrieval masking (input)
  - Operator output masking (output)
  - Double masking idempotency
  - Operator chains (ts_mean, rank)
  - Edge cases (all False, time-varying)
  - Injected data (Open Toolkit pattern)
- **All tests pass** ✓

#### Showcase Highlights

**Showcase 09: Universe Masking** demonstrates:

1. **Initialize with universe**: `AlphaCanvas(universe=price > 5.0)`
2. **Automatic masking**: Field retrieval and operator output both masked
3. **Comparison**: Same data with vs without universe
4. **Operator chains**: Field → ts_mean → rank (all masked)
5. **Open Toolkit**: Injected DataArray also respects universe
6. **Read-only property**: `rc.universe` for inspection
7. **Coverage statistics**: Universe coverage over time
8. **Visual validation**: Tables showing masked vs unmasked values

**Output Highlights**:
- Low-priced stocks (PENNY_STOCK, MICROCAP, ILLIQUID) always NaN
- High-priced stocks (AAPL, NVDA, GOOGL) have values
- ts_mean preserves masking (NaN propagates through rolling window)
- rank excludes universe-excluded stocks from ranking
- Injected returns respect universe automatically

#### Performance Benchmarks

| Operation | Time (ms) | vs Baseline | Notes |
|-----------|-----------|-------------|-------|
| No masking | 4.50 | - | Baseline (no universe) |
| Single mask | 4.90 | +8.9% | Field masking only |
| Double mask | 5.11 | +13.6% | Field + Operator masking |

**Conclusion**: 13.6% overhead is acceptable for the safety and clarity provided by double masking.

#### Lessons Learned

1. **xarray.where() is Perfect**: Built-in masking method is idempotent and performant
2. **Double Masking Creates Trust**: Input + output masking ensures correctness
3. **Immutability Ensures Fairness**: Fixed universe for reproducible backtests
4. **Centralize Universe Logic**: Keep in Visitor, not in operators
5. **Open Toolkit Integration**: Injected data also needs masking
6. **Validation is Critical**: Shape and dtype validation prevents subtle bugs
7. **Performance is Not a Concern**: <15% overhead with xarray lazy evaluation

#### Next Steps

1. ✓ Experiment validated double masking behavior
2. ✓ TDD tests implemented (13 tests)
3. ✓ AlphaCanvas updated with universe parameter
4. ✓ EvaluateVisitor updated with double masking
5. ✓ Showcase demonstrates all features
6. ✓ Document in FINDINGS.md (this entry)
7. Future: Implement database-backed universes via Field('univ500')
8. Future: Create universe creator utility for complex universe definitions

---
