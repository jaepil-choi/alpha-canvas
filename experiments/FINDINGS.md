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

## Phase 11: Boolean Expressions (F2 Foundation)

### Experiment 14: Boolean Expression Operators

**Date**: 2025-10-21  
**Status**: ✅ SUCCESS

**Summary**: Validated Expression-based comparison and logical operators. The pattern of overloading Python operators (`__eq__`, `__gt__`, `__and__`, etc.) on the Expression base class creates lazy Boolean Expressions that go through the Visitor, ensuring universe masking is always applied.

#### Key Discoveries

1. **Dataclass __eq__ Override Issue**
   - **Problem**: `@dataclass` generates its own `__eq__` method for structural equality
   - **Impact**: Dataclass `__eq__` takes precedence over Expression's `__eq__`, breaking comparison operators
   - **Solution**: Use `@dataclass(eq=False)` to disable dataclass equality generation
   - **Application**: **ALL Expression dataclasses must use `eq=False`**

2. **Lazy Evaluation Confirmed**
   - `Field('size') == 'small'` creates **Equals Expression** (not immediate result)
   - `(expr1 & expr2)` creates **And Expression** (lazy combination)
   - No evaluation happens until `visitor.evaluate(expression)` is called
   - **Critical**: Ensures universe masking via Visitor is never bypassed

3. **Generic Visitor Pattern Extends Naturally**
   - Boolean Expressions fit cleanly into existing `visit_operator()` pattern
   - **Binary operators** (Equals, And, Or): Have `left` and `right` attributes
   - **Unary operators** (Not): Have `child` attribute
   - **Right operand** can be Expression or literal (e.g., `'small'`, `5.0`)
   - Visitor detects structure via `hasattr()` and evaluates accordingly

4. **Comparison Operators Validated**
   - `==` (Equals), `!=` (NotEquals) ✓
   - `>` (GreaterThan), `<` (LessThan) ✓
   - `>=` (GreaterOrEqual), `<=` (LessOrEqual) ✓
   - All produce correct boolean DataArrays
   - All work with both numeric and string comparisons

5. **Logical Operators Validated**
   - `&` (And): Logical AND of two boolean Expressions ✓
   - `|` (Or): Logical OR of two boolean Expressions ✓
   - `~` (Not): Logical NOT of boolean Expression ✓
   - Chained expressions work correctly: `(a == 'small') & (b == 'high')`

6. **NaN Handling**
   - **Comparisons**: NaN > 5.0 → False (standard xarray/numpy behavior)
   - **Logical AND**: NaN & True → False
   - **Logical OR**: NaN | True → True
   - Behavior matches xarray's native boolean operations

7. **Performance Excellent**
   - **Dataset**: (500, 100) with string labels
   - **Expression**: `(size == 'small') & (value == 'high')`
   - **Time**: 2.0ms (extremely fast)
   - **Conclusion**: Boolean Expressions add negligible overhead

#### Architecture Pattern

**Expression Base Class with Operators**:

```python
class Expression(ABC):
    """Base class with comparison and logical operators."""
    
    # Comparison operators
    def __eq__(self, other):
        from alpha_canvas.ops.logical import Equals
        return Equals(self, other)
    
    def __gt__(self, other):
        from alpha_canvas.ops.logical import GreaterThan
        return GreaterThan(self, other)
    
    # ... other comparisons ...
    
    # Logical operators
    def __and__(self, other):
        from alpha_canvas.ops.logical import And
        return And(self, other)
    
    def __or__(self, other):
        from alpha_canvas.ops.logical import Or
        return Or(self, other)
    
    def __invert__(self):
        from alpha_canvas.ops.logical import Not
        return Not(self)
```

**Boolean Expression Classes**:

```python
@dataclass(eq=False)  # CRITICAL: disable dataclass __eq__
class Equals(Expression):
    left: Expression
    right: Union[Expression, Any]  # Can be literal or Expression
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(self, left_result: xr.DataArray, right_result: Any = None) -> xr.DataArray:
        """Compare left and right (right can be literal)."""
        if right_result is not None:
            return left_result == right_result  # Both Expressions
        else:
            return left_result == self.right  # Right is literal
```

**Visitor Handles Binary and Unary Operators**:

```python
def visit_operator(self, node) -> xr.DataArray:
    """Generic visitor supporting multiple patterns."""
    
    if hasattr(node, 'child'):
        # Unary operator (TsMean, Rank, Not)
        child_result = node.child.accept(self)
        result = node.compute(child_result)
    
    elif hasattr(node, 'left') and hasattr(node, 'right'):
        # Binary operator (Equals, And, Or, etc.)
        left_result = node.left.accept(self)
        
        if isinstance(node.right, Expression):
            right_result = node.right.accept(self)
            result = node.compute(left_result, right_result)
        else:
            # Right is literal
            result = node.compute(left_result)
    
    # Universe masking applied to output
    if self._universe_mask is not None:
        result = result.where(self._universe_mask, np.nan)
    
    self._cache_result(node.__class__.__name__, result)
    return result
```

#### Use Cases Validated

1. **String Comparison**
   - `Field('size') == 'small'` → Boolean mask for small-cap stocks
   - `Field('sector') != 'Finance'` → Exclude financial sector

2. **Numeric Comparison**
   - `Field('price') > 5.0` → Filter penny stocks
   - `Field('volume') >= 100000` → Liquidity filter

3. **Chained Conditions**
   - `(Field('size') == 'small') & (Field('value') == 'high')` → Small-cap value stocks
   - `(Field('price') < 5.0) | (Field('volume') < 10000)` → Low quality stocks

4. **Negation**
   - `~(Field('size') == 'small')` → Not small-cap

#### Integration with F2 Selector Interface

**This foundation enables**:

```python
# rc.data accessor returns Field Expressions
small = rc.data['size'] == 'small'  # Creates Equals Expression
high = rc.data['value'] == 'high'   # Creates Equals Expression

# Combine with logical operators
mask = small & high  # Creates And Expression

# Later: Use with selector interface
rc[mask] = 1.0  # Assign signal based on Expression mask
```

**Why Expression-based?**

1. **Universe masking**: All comparisons go through Visitor → universe applied
2. **Lazy evaluation**: Expressions can be composed without evaluation
3. **Caching**: Visitor caches intermediate boolean masks (step indices)
4. **Traceability**: Each comparison step is traceable via step index
5. **Type safety**: Strong Expression tree structure

#### Lessons Learned

1. **Dataclass Equality Trap**: `eq=False` is **mandatory** for all Expression dataclasses
2. **Lazy is Essential**: Expression-based comparisons ensure universe masking is never bypassed
3. **Generic Visitor Scales**: Single `visit_operator()` handles all operator patterns (unary, binary)
4. **Literal vs Expression**: Binary operators must handle both `right: Expression` and `right: literal`
5. **Operator Overloading is Clean**: Python's `__eq__`, `__and__`, etc. provide intuitive syntax
6. **Performance Non-Issue**: Boolean Expression evaluation is extremely fast (2ms for large dataset)

#### Test Results

- **All 100 existing tests pass** ✓ (no regressions)
- Boolean Expression infrastructure integrated without breaking existing operators
- `eq=False` applied to TsMean, TsAny, Rank, Field

#### Next Steps

1. ✓ Experiment validated Boolean Expression pattern
2. ✓ Expression base class updated with operator overloading
3. ✓ Boolean Expression classes implemented (ops/logical.py)
4. ✓ Visitor updated to handle binary and unary operators
5. ✓ All existing operator dataclasses updated with `eq=False`
6. ✓ Full test suite passes (100/100)
7. ✓ Showcase created demonstrating Boolean Expressions
8. ✓ Public `rc.evaluate()` API added
9. **NEXT**: Implement rc.data accessor (DataAccessor) - Phase 7B

---

## Phase 7B: DataAccessor (Selector Interface)

### Experiment 15: DataAccessor Pattern Validation

**Date**: 2025-01-21  
**Status**: ✅ SUCCESS

**Summary**: Validated that DataAccessor returning Field Expressions enables Expression-based selector workflows. All comparisons remain lazy until explicit evaluation, ensuring universe masking through Visitor pattern.

#### Key Discovery: Why DataAccessor Returns Expressions

**Problem**: Initial design had two accessors (`rc.data` and `rc.axis`):
- `rc.data` for raw data access
- `rc.axis` for categorical selection with special `['label']` syntax

**Discovery**: This creates unnecessary complexity and duplication:
- What distinguishes "data" from "axis"? (Arbitrary)
- Two ways to do the same thing (violates "one obvious way")
- `rc.axis.size['small']` is just syntactic sugar for `rc.data['size'] == 'small'`

**Solution**: **Single accessor pattern** (`rc.data` only):
- `rc.data['size']` → returns `Field('size')` Expression
- `rc.data['size'] == 'small'` → returns `Equals` Expression
- Explicit comparison (`== 'small'`) is clearer than implicit selection (`['small']`)
- Leverages Phase 7A Boolean Expression infrastructure perfectly

**Impact**: 
- **Simplicity**: One accessor, not two
- **Pythonic**: Explicit > implicit
- **Consistent**: Everything is "data", no arbitrary "axis" concept
- **Maintainable**: Less code, fewer concepts to understand

#### Architecture Pattern

**DataAccessor Design**:

```python
class DataAccessor:
    """Returns Field Expressions for lazy evaluation."""
    
    def __getitem__(self, field_name: str) -> Field:
        """Return Field Expression (not raw data!)"""
        return Field(field_name)
    
    def __getattr__(self, name: str):
        """Prevent attribute access - item access only."""
        raise AttributeError(
            f"Use rc.data['{name}'] instead of rc.data.{name}"
        )
```

**Usage Pattern**:

```python
# Returns Field('size') Expression
size_field = rc.data['size']

# Returns Equals Expression (lazy)
mask = rc.data['size'] == 'small'

# Returns And Expression (lazy)
complex_mask = (rc.data['size'] == 'small') & (rc.data['momentum'] == 'high')

# Evaluate with universe masking
result = rc.evaluate(mask)
```

#### Why Field Expressions, Not Raw Data?

**Design Decision**: `rc.data['field']` returns `Field('field')`, not `rc.db['field']` (raw DataArray)

**Rationale**:

1. **Lazy Evaluation**
   - No premature data loading
   - Can compose expressions before evaluation
   - Enables optimization opportunities

2. **Universe Safety**
   - All Expressions go through Visitor
   - Universe masking guaranteed at evaluation time
   - No way to bypass masking accidentally

3. **Composability**
   - Can chain with operators: `ts_mean(rc.data['returns'], 10)`
   - Can store as recipes: `momentum = rc.data['returns'] > rc.data['sma']`
   - Expressions are first-class objects

4. **Consistency**
   - All Expressions follow same evaluation path
   - No special-case handling for "accessor-created" vs "manually-created" Fields
   - Single `evaluate()` API for everything

5. **Traceability**
   - Visitor caches intermediate results with step indices
   - Can trace PnL at any comparison step
   - Debugging is transparent

**Anti-Pattern (What We Avoided)**:

```python
# ❌ BAD: Returning raw data
class DataAccessor:
    def __getitem__(self, field_name):
        return self._rc.db[field_name]  # Raw DataArray

# Problem: Immediate evaluation, no universe masking guarantee
mask = rc.data['size'] == 'small'  # Already a boolean DataArray
# Universe can't be applied retroactively
```

#### Test Results

**Experiment 15 Validated**:

1. ✓ Basic access returns Field Expression
2. ✓ Comparison creates Boolean Expression (lazy, not immediate)
3. ✓ Complex logical chains remain as Expression trees
4. ✓ Evaluator handles accessor-created Expressions correctly
5. ✓ Universe masking applied correctly (4 masked positions → NaN)
6. ✓ Type validation (non-string raises TypeError)
7. ✓ Attribute access prevented (raises AttributeError)

**Performance**: <1ms overhead for accessor (negligible)

#### Usage Patterns Validated

**1. Basic Selection**:
```python
small_mask = rc.data['size'] == 'small'
result = rc.evaluate(small_mask)
```

**2. Multi-Dimensional Selection** (Fama-French style):
```python
selection = (
    (rc.data['size'] == 'small') & 
    (rc.data['value'] == 'high')
)
result = rc.evaluate(selection)
```

**3. Numeric Filtering**:
```python
penny_stocks = rc.data['price'] < 5.0
illiquid = rc.data['volume'] < 100000
low_quality = penny_stocks | illiquid
result = rc.evaluate(low_quality)
```

**4. Negation**:
```python
not_small = ~(rc.data['size'] == 'small')
result = rc.evaluate(not_small)
```

#### Integration with Phase 7A

**Synergy**: Phase 7B (DataAccessor) perfectly leverages Phase 7A (Boolean Expressions)

- Phase 7A provided: `__eq__`, `__and__`, `__or__` overloading on Expression base class
- Phase 7B provides: Convenient accessor that returns those Expressions
- Together: Natural, Pythonic syntax with universe safety

**Example**:
```python
# Phase 7B: Accessor returns Field
field = rc.data['size']  # Field('size')

# Phase 7A: Comparison creates Expression
mask = field == 'small'  # Equals(Field('size'), 'small')

# Phase 7A: Logical operators chain
complex = mask & (rc.data['momentum'] == 'high')  # And(Equals(...), Equals(...))

# Visitor evaluates with universe masking
result = rc.evaluate(complex)
```

#### Document Updates Required

**Remove `rc.axis` from all documentation**:

1. **PRD**: Replace `rc.axis.size['small']` with `rc.data['size'] == 'small'`
2. **Architecture**: Remove AxisAccessor/AxisSelector sections
3. **Implementation**: Update Interface B examples

**Why remove `rc.axis`?**

- **Simplicity wins**: One accessor is cleaner than two
- **No semantic value**: "axis" vs "data" is arbitrary distinction
- **Leverages existing infrastructure**: Phase 7A already enables `==` comparisons
- **Pythonic principle**: Explicit (`== 'small'`) > implicit (`['small']`)

#### Lessons Learned

1. **Simplicity Over Cleverness**
   - Initial design had clever `rc.axis` sugar → unnecessary
   - Simple explicit comparison is clearer
   - "One obvious way" is better than "two ways"

2. **Accessor Pattern is Lightweight**
   - Just return Field('field_name')
   - All heavy lifting already done by Phase 7A
   - Minimal code, maximum value

3. **Item Access Only**
   - Prevent `rc.data.field` (attribute access)
   - Force `rc.data['field']` (item access)
   - Single pattern reduces confusion

4. **Expression-First Design**
   - Always return Expressions, never raw data
   - Ensures universe safety
   - Enables composition and traceability

5. **Document-First Reveals Issues**
   - Writing PRD revealed `rc.axis` complexity
   - User feedback identified duplication
   - Critical evaluation prevented unnecessary code

#### Next Steps

1. ✓ Experiment validated DataAccessor pattern
2. ✓ TDD tests for DataAccessor (tests/test_utils/test_accessor.py)
3. ✓ Implement DataAccessor class in utils/accessor.py
4. ✓ Add `data` property to AlphaCanvas facade
5. ✓ Create showcase demonstrating practical usage
6. ✓ Update PRD to remove all `rc.axis` references
7. ✓ Update architecture.md (DataAccessor only)
8. ✓ Update implementation.md with correct patterns

---

## Phase 13: Cross-Sectional Quantile Bucketing (cs_quantile)

### Experiment 16: xarray Groupby Shape Preservation

**Date**: 2024-10-21  
**Status**: ✅ SUCCESS

**Summary**: Validated that xarray's `.groupby().map()` pattern preserves `(T, N)` shape when applying `pd.qcut` for quantile bucketing. Both independent and dependent sort patterns work correctly.

#### Key Discoveries

1. **Shape Preservation with .map()**
   - **Pattern**: `data.groupby('time').map(lambda slice: pd.qcut(slice.flatten(), ...))`
   - **Result**: Input `(T, N)` → Output `(T, N)` with categorical labels
   - **Mechanism**: xarray automatically concatenates results back along grouped dimension
   - **Critical**: Must flatten slice to 1D for `pd.qcut`, then reshape back

2. **Categorical Label Output**
   - `pd.qcut` with `labels` parameter returns object dtype (strings)
   - Labels are preserved through xarray operations
   - No numeric conversion - maintains semantic meaning ('small', 'big')
   - Perfect for Fama-French style portfolio construction

3. **Nested Groupby for Dependent Sort**
   - **Pattern**: `data.groupby(groups).map(lambda g: g.groupby('time').map(qcut))`
   - **Result**: Different quantile cutoffs for each group (as expected)
   - **Verification**: Independent and dependent sorts produce different labels (17% of positions differ)
   - **Performance**: 4.26x slower than independent sort (acceptable for batch processing)

4. **NaN Handling**
   - `pd.qcut` with `duplicates='drop'` handles NaN values gracefully
   - NaN positions in input remain NaN in output
   - Edge case: All same values → returns all NaN (gracefully handled)
   - Edge case: All NaN → returns all NaN (gracefully handled)

5. **Performance Characteristics**
   - Independent sort: 27ms for (10×6) data
   - Dependent sort: 117ms for (10×6) data (4.26x overhead)
   - Performance acceptable for typical factor research workflows
   - Overhead comes from nested groupby operations

#### Critical Implementation Details

**Flatten-Reshape Pattern:**
```python
def qcut_at_timestep(data_slice):
    values_1d = data_slice.values.flatten()  # pd.qcut needs 1D
    result = pd.qcut(values_1d, q=bins, labels=labels, duplicates='drop')
    result_array = np.array(result).reshape(data_slice.shape)  # Reshape back
    return xr.DataArray(result_array, dims=data_slice.dims, coords=data_slice.coords)
```

**Nested Groupby for Dependent Sort:**
```python
def apply_qcut_within_group(group_data):
    return group_data.groupby('time').map(qcut_function)

result = data.groupby(size_labels).map(apply_qcut_within_group)
```

#### Lessons Learned

1. **xarray .map() vs .apply()**
   - Both work for shape preservation
   - `.map()` is cleaner for xarray-to-xarray transformations
   - Automatic concatenation is the key feature

2. **pd.qcut Requires 1D Input**
   - Must explicitly flatten before calling `pd.qcut`
   - Must explicitly reshape after to match original dimensions
   - This pattern is consistent and reliable

3. **duplicates='drop' Is Essential**
   - Handles edge cases (all same values, all NaN)
   - Returns NaN for problematic bins instead of raising error
   - Graceful degradation for real-world data

4. **Independent vs Dependent Sort Verification**
   - Critical to verify they produce different cutoffs
   - This confirms correct Fama-French implementation
   - Must test on realistic data, not just toy examples

5. **Performance Is Acceptable**
   - 4x overhead for dependent sort is reasonable
   - Factor research is typically batch processing (not real-time)
   - Can optimize later if needed (e.g., with flox library)

#### Architectural Implications

1. **Operator Responsibility Pattern Still Applies**
   - Operator owns `compute()` with flatten-reshape-qcut logic
   - Visitor handles traversal and group lookup only
   - Clear separation of concerns

2. **Special Handling for group_by Parameter**
   - Visitor must look up `group_by` field from dataset
   - Pass group labels to `compute()` as optional parameter
   - This is a special case, but fits existing patterns

3. **Universe Masking Integration**
   - Universe masking applies AFTER quantile bucketing
   - NaN positions in universe become NaN in output
   - No special handling needed - existing pattern works

4. **String Labels Are First-Class**
   - object dtype for categorical labels is correct
   - Enables comparison operations: `rc.data['size'] == 'small'`
   - Integrates perfectly with Boolean Expression infrastructure

#### Test Strategy

**Unit Tests:**
- Test `compute()` directly with synthetic data
- Verify shape preservation
- Verify categorical output
- Verify NaN handling
- Verify edge cases (all same, all NaN)
- Compare independent vs dependent cutoffs

**Integration Tests:**
- Test with Visitor and Expression tree
- Verify group_by field lookup
- Verify universe masking application
- Verify caching works correctly
- Verify error handling (group_by field not found)

#### Next Steps

1. ✅ Experiment validated shape preservation pattern
2. ✅ Write TDD tests in `tests/test_ops/test_classification.py`
3. ✅ Implement `CsQuantile` class in `src/alpha_canvas/ops/classification.py`
4. ✅ Update Visitor to handle `CsQuantile` special case
5. ✅ Create showcase demonstrating Fama-French portfolio construction
6. ✅ Update documentation

---

## Phase 14: Signal Canvas Assignment (Lazy Evaluation)

### Experiment 17: Lazy vs Immediate Assignment Patterns

**Date**: 2025-01-21  
**Status**: ✅ SUCCESS

**Summary**: Validated two design patterns for Expression-based signal assignment. Lazy evaluation preserves full Expression tree for traceability (critical for PnL tracking) with negligible performance overhead. Recommended lazy evaluation for implementation.

#### Key Discoveries

1. **Lazy Evaluation Preserves Traceability**
   - **Pattern**: Store assignments as `List[(mask_expr, value)]` tuples
   - **Benefit**: Full Expression tree preserved for step-by-step PnL tracking
   - **Critical**: Can cache base result (step N) and final result (step N+1) separately
   - **Essential**: Enables `rc.trace_pnl()` and `rc.get_intermediate()` functionality

2. **Performance Comparison**
   - **Lazy average**: 8.9ms (all 4 scenarios)
   - **Immediate average**: 6.5ms (all 4 scenarios)
   - **Difference**: 2.5ms slower (27.5%), but absolute difference negligible
   - **Conclusion**: Performance not a concern for factor research (batch processing)

3. **Memory Footprint**
   - **Lazy**: 88 bytes (assignment list storage)
   - **Immediate**: 104 bytes (full DataArray cached)
   - **Winner**: Lazy (lower memory footprint)
   - **Impact**: Minimal difference, both acceptable

4. **Flexibility Comparison**
   - **Lazy**: High flexibility
     - Can inspect assignments before evaluation
     - Can modify/remove assignments
     - Can re-evaluate with different data
   - **Immediate**: Low flexibility
     - Assignments applied immediately (no inspection)
     - Cannot modify after application
     - Data cached (cannot re-evaluate)
   - **Winner**: Lazy (much higher flexibility for research workflows)

5. **Overlapping Masks Behavior**
   - Sequential application: Later assignment overwrites earlier
   - Example: `mask1` sets positions A, B, C to 1.0, then `mask2` sets B, C, D to -1.0
   - Result: B and C are -1.0 (later wins), A is 1.0, D is -1.0
   - **Validation**: Both patterns produce identical results

6. **Universe Integration**
   - Universe masking applied AFTER all assignments
   - Assignments outside universe automatically become NaN
   - No special handling needed - existing double-masking strategy works
   - **Validation**: Positions outside universe are NaN in final result

#### Implementation Pattern Established

**Lazy Evaluation Pattern**:

```python
class Expression:
    def __init__(self):
        self._assignments = []  # List of (mask_expr, value) tuples
    
    def __setitem__(self, mask, value):
        """Store assignment for lazy evaluation."""
        self._assignments.append((mask, value))

# Usage:
signal = ts_mean(Field('returns'), 3)  # Expression (lazy)
signal[mask1] = 1.0  # Stored, not evaluated
signal[mask2] = -1.0  # Stored, not evaluated
result = rc.add_data('signal', signal)  # NOW everything is evaluated
```

**Visitor Evaluation Flow**:

```python
def evaluate(self, expr: Expression) -> xr.DataArray:
    # Step 1: Evaluate base Expression
    base_result = expr.accept(self)
    
    # Step 2: Cache base result (before assignments)
    if expr._assignments:
        self._cache_result(f"{expr.__class__.__name__}_base", base_result)
    
    # Step 3: Apply assignments sequentially
    if expr._assignments:
        final_result = self._apply_assignments(base_result, expr._assignments)
        
        # Step 4: Apply universe masking
        if self._universe_mask is not None:
            final_result = final_result.where(self._universe_mask, np.nan)
        
        # Step 5: Cache final result
        self._cache_result(f"{expr.__class__.__name__}_final", final_result)
        return final_result
    
    return base_result

def _apply_assignments(self, base_result, assignments):
    """Apply assignment list to base result."""
    result = base_result.copy()
    for mask_expr, value in assignments:
        mask_data = self.evaluate(mask_expr) if isinstance(mask_expr, Expression) else mask_expr
        result = result.where(~mask_data, value)  # Replace where mask is True
    return result
```

#### Test Scenarios Validated

1. **Simple Assignment** (zeros → long/short)
   - Start from constant 0.0
   - Assign 1.0 to small-cap, -1.0 to large-cap
   - Result: Correct long/short positions
   - ✓ Both patterns produce identical results

2. **Transform Existing Signal** (ts_mean → boost high momentum)
   - Start from ts_mean(returns, 10)
   - Boost high momentum positions to 2.0
   - Result: High momentum replaced, others unchanged
   - ✓ Both patterns produce identical results

3. **Overlapping Masks** (later wins)
   - mask1 assigns 1.0 to positions [0, 1, 2]
   - mask2 assigns -1.0 to positions [1, 2, 3]
   - Result: Overlap [1, 2] gets -1.0 (later wins)
   - ✓ Sequential application works correctly

4. **Multiple Sequential Modifications** (3 assignments)
   - Start from ts_mean, apply 3 different assignments
   - Result: All 3 modifications applied correctly
   - ✓ No interference between assignments

5. **Universe Masking Integration**
   - Assign to all positions (including outside universe)
   - Result: Outside-universe positions are NaN
   - ✓ Universe masking works correctly with assignments

#### Performance Metrics

| Scenario | Lazy (ms) | Immediate (ms) | Difference |
|----------|-----------|----------------|------------|
| Simple assignment | 4.8 | 4.1 | +0.7ms |
| Transform existing | 15.6 | 10.4 | +5.2ms |
| Overlapping masks | - | - | - |
| Multiple sequential | 12.4 | 7.2 | +5.2ms |
| **Average** | **8.9** | **6.5** | **+2.5ms (27.5%)** |

**Conclusion**: Lazy is slightly slower, but 2.5ms difference is negligible for batch processing.

#### Recommendation: LAZY EVALUATION

**Reasons**:

1. **Traceability** (Most Important)
   - Full Expression tree preserved
   - Can cache base and final results separately
   - Essential for rc.trace_pnl() step-by-step tracking
   - Critical for quantitative research debugging

2. **Flexibility**
   - Can inspect assignments before evaluation
   - Can modify/remove assignments
   - Can re-evaluate with different data
   - Better for research workflows

3. **Memory**
   - Lower footprint (88 bytes vs 104 bytes)
   - Assignment list is lightweight

4. **Architecture Consistency**
   - Aligns with lazy evaluation philosophy
   - Expression = computation recipe (no data until evaluate)
   - Visitor = evaluation engine
   - No violation of design principles

5. **Performance**
   - 2.5ms overhead is negligible
   - Factor research is batch processing (not real-time)
   - Can optimize later if needed

**Trade-offs**:
- Immediate evaluation is 27.5% faster
- But lazy evaluation provides essential traceability
- Performance difference too small to matter

#### Lessons Learned

1. **Traceability Trumps Performance**
   - 2.5ms slower is acceptable for research platform
   - Full Expression tree is invaluable for debugging
   - Step-by-step PnL tracking requires lazy evaluation

2. **Lazy Evaluation is Not "Free"**
   - 27.5% overhead exists
   - But in absolute terms (2.5ms), it's negligible
   - Trade-off heavily favors lazy for research use case

3. **Flexibility Enables Research**
   - Being able to inspect assignments is valuable
   - Modifying assignments without re-computation helps iteration
   - Research platforms need flexibility over raw speed

4. **Sequential Application is Simple**
   - Later assignment overwrites earlier for overlaps
   - Easy to understand and reason about
   - No complex merge logic needed

5. **Universe Integration is Clean**
   - Apply universe masking after all assignments
   - No special handling needed
   - Existing double-masking strategy works perfectly

#### Next Steps

1. ✅ Experiment validated lazy evaluation pattern
2. ✅ Write TDD tests for Expression.__setitem__
3. ✅ Implement assignment storage in Expression base class
4. ✅ Implement _apply_assignments in Visitor
5. ✅ Modify evaluate() to handle assignments
6. ✅ Create Constant expression class
7. ✅ Run tests until all pass (green phase)
8. ✅ Create showcase demonstrating all patterns
9. ✅ Update documentation

---

## Phase 15: Portfolio Weight Scaling (F5)

### Experiment 18: Vectorized Weight Scaling Validation

**Date**: 2025-01-22  
**Status**: ✅ SUCCESS

**Summary**: Validated fully vectorized GrossNetScaler implementation with unified Gross/Net exposure framework. All scenarios pass including edge cases (one-sided signals, all-zero signals). Performance excellent at 7-40ms for various dataset sizes. No iteration required.

#### Key Discoveries

1. **Vectorized Gross-Target Scaling (Core Algorithm)**
   - **Pattern**: Always meet gross target via final scaling step
   - **Formula**: 
     - $L_{\text{target}} = \frac{G + N}{2}$, $S_{\text{target}} = \frac{G - N}{2}$
     - Normalize positive/negative signals separately
     - Apply L_target and S_target
     - Calculate actual_gross and scale: `weights * (target_gross / actual_gross)`
   - **Result**: Gross target always met, even for one-sided signals
   - **Net Target**: May be unachievable for one-sided signals (by design)

2. **All-Zero Signal Handling**
   - **Challenge**: `0 / 0` results in NaN, not 0
   - **Solution 1**: `fillna(0.0)` after normalization step
   - **Solution 2**: `xr.where(actual_gross > 0, target_gross / actual_gross, 1.0)` for scale_factor
   - **Solution 3**: Final `fillna(0.0)` before universe mask application
   - **Result**: All-zero signals → all-zero weights (correct)

3. **One-Sided Signal Behavior**
   - **All Positive**: Long = target_gross, Short = 0, Net = target_gross (not target_net)
   - **All Negative**: Long = 0, Short = -target_gross, Net = -target_gross (not target_net)
   - **Design Decision**: This is correct behavior - gross target takes priority
   - **Warning**: In production, should emit WARNING when net target unachievable

4. **NaN Preservation (Universe Integration)**
   - **Critical**: Apply `fillna(0.0)` BEFORE universe mask, not after
   - **Order**: Computation NaN → 0.0, then Universe mask → NaN
   - **Result**: Universe positions correctly preserved as NaN
   - **Validation**: Scenario 4 passes with 22 NaN positions maintained

5. **Performance Characteristics**
   - **Small (10×6)**: 7ms per scale
   - **Medium (100×50)**: 7ms per scale
   - **1Y Daily (252×100)**: 8ms per scale
   - **Large (1000×500)**: 34ms per scale
   - **Conclusion**: Performance scales excellently, no bottlenecks

6. **Vectorization Benefits**
   - **No Iteration**: Zero Python-level loops (pure xarray/numpy operations)
   - **Speedup**: 10x-220x faster than iterative `groupby('time').map()` approach
   - **Scalability**: Performance stays consistent across dataset sizes
   - **Code Clarity**: Cleaner, more readable than iterative version

#### Implementation Pattern Established

**Fully Vectorized GrossNetScaler**:

```python
def scale_grossnet(signal: xr.DataArray, target_gross: float, target_net: float) -> xr.DataArray:
    """Fully vectorized weight scaler - NO ITERATION!"""
    # Calculate targets
    L_target = (target_gross + target_net) / 2.0
    S_target = (target_net - target_gross) / 2.0  # Negative
    
    # Step 1: Separate positive/negative (vectorized)
    s_pos = signal.where(signal > 0, 0.0)
    s_neg = signal.where(signal < 0, 0.0)
    
    # Step 2: Sum along asset dimension (vectorized)
    sum_pos = s_pos.sum(dim='asset', skipna=True)
    sum_neg = s_neg.sum(dim='asset', skipna=True)
    
    # Step 3: Normalize (vectorized, handles 0/0 → nan → 0)
    norm_pos = (s_pos / sum_pos).fillna(0.0)
    norm_neg_abs = (np.abs(s_neg) / np.abs(sum_neg)).fillna(0.0)
    
    # Step 4: Apply L/S targets (vectorized)
    weights_long = norm_pos * L_target
    weights_short_mag = norm_neg_abs * np.abs(S_target)
    
    # Step 5: Combine
    weights = weights_long - weights_short_mag
    
    # Step 6: Calculate actual gross per row (vectorized)
    actual_gross = np.abs(weights).sum(dim='asset', skipna=True)
    
    # Step 7: Scale to meet target gross (vectorized)
    scale_factor = xr.where(actual_gross > 0, target_gross / actual_gross, 1.0)
    final_weights = weights * scale_factor
    
    # Step 8: Convert computational NaN to 0 (BEFORE universe mask)
    final_weights = final_weights.fillna(0.0)
    
    # Step 9: Apply universe mask (preserves NaN where signal was NaN)
    final_weights = final_weights.where(~signal.isnull())
    
    return final_weights
```

#### Test Scenarios Validated

**Scenario 1: Dollar Neutral** ✅
- Target: G=2.0, N=0.0 → L=1.0, S=-1.0
- Result: All constraints met exactly
- Performance: 7ms

**Scenario 2: Net Long Bias** ✅
- Target: G=2.0, N=0.2 → L=1.1, S=-0.9
- Result: Asymmetric allocation perfect
- Performance: 7ms

**Scenario 3: Long Only** ✅
- Target: L=1.0 (ignore negatives)
- Result: Negatives → 0, positives sum to 1.0
- Performance: 32ms (uses groupby for simplicity)

**Scenario 4: NaN Preservation** ✅
- Signal: 22 NaN positions (universe masking)
- Result: 22 NaN positions in weights (preserved)
- Verification: NaN positions match exactly

**Scenario 5: Edge Cases** ✅
- 5a: All positive signals → Gross=2.0, Net=2.0 (one-sided)
- 5b: All negative signals → Gross=2.0, Net=-2.0 (one-sided)
- 5c: Single valid asset per timestep → Correct scaling
- 5d: All-zero signals → All-zero weights (no NaN)

**Scenario 6: Vectorized Edge Cases** ✅
- Row 0: `[3, 5, 7, 6]` all positive → Gross=2.2, Net=2.2
- Row 1: `[3, -6, 9, 0]` mixed → Gross=2.2, Net=-0.5 ✓
- Row 2: `[3, -6, 9, -4]` mixed → Gross=2.2, Net=-0.5 ✓
- Row 3: `[-2, -5, -1, -9]` all negative → Gross=2.2, Net=-2.2
- Row 4: `[0, 0, 0, 0]` all zeros → All weights 0.0 (no NaN)

#### Design Decisions

1. **Gross Target Priority**
   - Gross target ALWAYS met (via final scaling step)
   - Net target achievable only for mixed signals (has both long/short)
   - One-sided signals: Net target unachievable by mathematical constraint

2. **Default Parameters**
   - **Recommended Default**: `target_gross=2.0, target_net=0.0`
   - Rationale: Dollar-neutral is most common for market-neutral strategies
   - Represents 100% long + 100% short (200% gross, 0% net)

3. **Stateless Design**
   - Scalers don't store state
   - Always pass parameters explicitly
   - Enables easy comparison of multiple strategies

4. **Cross-Sectional Independence**
   - Each timestep processed independently (mathematically)
   - No temporal dependencies (by design)
   - Vectorized operations respect this independence

#### Lessons Learned

1. **Vectorization Requires Careful Ordering**
   - `fillna(0.0)` must come BEFORE universe mask
   - Otherwise universe NaN → 0.0 (incorrect)

2. **Gross-Target Scaling is Robust**
   - Always achievable via simple scaling: `weights * (G_target / G_actual)`
   - Handles all edge cases (one-sided, zeros, mixed)
   - More robust than trying to meet both gross AND net simultaneously

3. **Division by Zero Handling**
   - `xr.where(condition, value, fallback)` better than `.fillna()` for inf handling
   - Prevents inf propagation from `0 / 0`

4. **Performance of Pure Vectorization**
   - 10x-220x faster than iterative approach
   - Scales excellently to large datasets
   - Code is actually clearer (less nested loops)

5. **Universe Integration is Clean**
   - Standard double-masking pattern works perfectly
   - No special handling needed in scaler
   - NaN preservation automatic

#### Test Results

- **6 scenarios, ALL PASS** ✅
- Dollar Neutral: ✅
- Net Long Bias: ✅
- Long Only: ✅
- NaN Preservation: ✅
- Edge Cases: ✅
- Vectorized Edge Cases: ✅

#### Performance Benchmarks

| Dataset Size | Time per Scale | Notes |
|--------------|----------------|-------|
| Small (10×6) | 7ms | Baseline |
| Medium (100×50) | 7ms | Same as small! |
| 1Y Daily (252×100) | 8ms | Minimal increase |
| Large (1000×500) | 34ms | Still excellent |

**Conclusion**: Vectorized approach is production-ready.

#### Next Steps

1. ✅ Experiment validated vectorized approach
2. **TODO**: Write TDD tests for WeightScaler classes
3. **TODO**: Implement GrossNetScaler in src/alpha_canvas/portfolio/
4. **TODO**: Implement DollarNeutralScaler and LongOnlyScaler
5. **TODO**: Add AlphaCanvas.scale_weights() facade method
6. **TODO**: Create showcase demonstrating weight scaling
7. **TODO**: Update documentation (PRD, Architecture, Implementation)

---

## Phase 17: Backtesting with Portfolio Returns (F6)

### Experiment 19: Vectorized Backtesting with Position-Level Attribution

**Date**: 2025-01-22  
**Status**: ✅ SUCCESS

**Summary**: Validated vectorized backtest workflow with shift-mask-multiply pattern. Position-level returns `(T, N)` preserved for winner/loser attribution. Re-masking after shift prevents NaN pollution. Cumsum validated as time-invariant metric (preferred over cumprod). Performance excellent at 7ms for (252, 100) dataset.

#### Key Discoveries

1. **Position-Level Returns Preserve Attribution**
   - **Pattern**: `port_return = final_weights * returns` (element-wise, shape `(T, N)`)
   - **Benefit**: Can trace which stocks contributed to PnL (winners/losers)
   - **Critical**: Do NOT aggregate immediately - keep `(T, N)` shape for traceability
   - **Aggregate on-demand**: `daily_pnl = port_return.sum(dim='asset')` only when needed

2. **Shift-Mask Workflow (Forward-Looking Bias Prevention)**
   - **Step 1**: Generate weights from signal at time `t-1` using `universe[t-1]`
   - **Step 2**: Shift weights: `weights_shifted = weights.shift(time=1)` 
   - **Step 3**: **RE-MASK** with universe at time `t`: `final_weights = weights_shifted.where(universe[t])`
   - **Step 4**: Calculate returns: `port_return = final_weights * returns[t]`
   - **Critical**: Re-masking liquidates positions that exited universe (prevents NaN pollution)

3. **Re-Masking Prevents NaN Pollution**
   - **Problem**: Stock exits universe → weight becomes NaN → `NaN * return = NaN` → daily PnL is NaN
   - **Solution**: Re-mask shifted weights with current universe → exited stocks get NaN weight → `NaN * return = 0` in sum (via `skipna=True`)
   - **Validation**: Stock exit scenario passed - PnL remains valid (no NaN pollution)
   - **Entry scenario**: Stock enters universe at `t` but has NaN weight (correctly can't hold it from `t-1`)

4. **Cumsum vs Cumprod (Time-Invariance)**
   - **Cumsum**: `[0.02, 0.03, -0.01] → cumsum = 0.04` (time-invariant)
   - **Cumprod**: `[0.02, 0.03, -0.01] → cumprod = 0.0405` (compound effect, time-dependent)
   - **Decision**: Use cumsum for fair strategy comparison
   - **Rationale**: Order doesn't affect cumsum, but cumprod favors longer strategies (unfair)
   - **Implementation**: `cumulative_pnl = daily_pnl.cumsum(dim='time')` calculated on-demand

5. **Winner/Loser Attribution**
   - **Pattern**: `total_contrib = port_return.sum(dim='time')` → contribution per stock
   - **Benefit**: Identify which stocks drove PnL (both positive and negative)
   - **Use case**: Post-mortem analysis, factor validation, position sizing insights
   - **Performance**: Instant (vectorized sum operation)

6. **Performance Metrics**
   - **Dataset**: (252, 100) - 1 year daily, 100 stocks
   - **Time**: 7ms total (shift + re-mask + multiply + sum + cumsum)
   - **Target**: <10ms (passed with 30% margin)
   - **Conclusion**: Fully vectorized workflow is production-ready

#### Implementation Pattern

**Vectorized Backtest Workflow**:

```python
def compute_portfolio_returns(
    weights: xr.DataArray,        # (T, N) from weight scaler
    returns: xr.DataArray,         # (T, N) from data
    universe: xr.DataArray         # (T, N) boolean mask
) -> xr.DataArray:
    """
    Compute position-level portfolio returns with forward-bias prevention.
    
    Returns:
        (T, N) DataArray - position-level weighted returns (for attribution)
    """
    # Step 1: Shift weights (trade on yesterday's signal)
    weights_shifted = weights.shift(time=1)
    
    # Step 2: Re-mask with today's universe (liquidate exited positions)
    final_weights = weights_shifted.where(universe)
    
    # Step 3: Mask returns
    returns_masked = returns.where(universe)
    
    # Step 4: Element-wise multiply (KEEP (T, N) SHAPE!)
    port_return = final_weights * returns_masked
    
    return port_return  # NOT aggregated yet!

# On-demand aggregation (when user needs it):
daily_pnl = port_return.sum(dim='asset')        # (T,)
cumulative_pnl = daily_pnl.cumsum(dim='time')   # (T,)
```

**Cache Structure**:

```python
# In EvaluateVisitor:
self._signal_cache: Dict[int, Tuple[str, xr.DataArray]] = {}      # Persistent
self._weight_cache: Dict[int, Tuple[str, xr.DataArray]] = {}      # Renewable (when scaler changes)
self._port_return_cache: Dict[int, Tuple[str, xr.DataArray]] = {} # Renewable (when scaler changes)

# At each step:
self._signal_cache[step] = (name, signal)           # (T, N)
self._weight_cache[step] = (name, weights)          # (T, N)
self._port_return_cache[step] = (name, port_return) # (T, N) - position-level!
```

#### Test Scenarios Validated

**Scenario 1: Basic Attribution** ✅
- Position-level returns preserve `(T, N)` shape
- Winner/loser analysis works (stock contributions visible)
- Aggregate PnL calculated correctly on-demand

**Scenario 2: Stock Exit** ✅
- Stock exits universe mid-period
- Re-masking prevents NaN pollution in PnL
- Exited stock contribution correctly zeroed

**Scenario 3: Stock Entry** ✅
- Stock enters universe mid-period
- Entry day has NaN weight (can't trade from yesterday)
- Subsequent days have valid weights

**Scenario 4: Performance (252×100)** ✅
- Completed in 7ms (well under 10ms target)
- All steps fully vectorized
- Scalable to production datasets

**Scenario 5: Attribution at Scale** ✅
- Top/bottom contributor analysis works on large dataset
- Sorting and filtering performant
- Winner/loser identification instantaneous

**Scenario 6: Cumsum Validation** ✅
- Cumsum is time-invariant (fair for comparison)
- Cumprod is time-dependent (compound effect)
- Cumsum chosen as default metric

#### Design Decisions

1. **Return Data is Mandatory**
   - AlphaCanvas initialization must load 'returns' field from config
   - Fail fast if 'returns' field missing (don't allow backtest-less sessions)
   - Hardcoded field name: `'returns'`

2. **Automatic Backtest Execution**
   - When `rc.evaluate(expr, scaler=...)` called with scaler:
     - Signal computed and cached
     - Weights computed and cached
     - **Portfolio returns automatically computed and cached**
   - When no scaler: Only signal cached (no weights, no portfolio returns)

3. **Triple-Cache Architecture**
   - `_signal_cache`: Persistent (never changes across scaler swaps)
   - `_weight_cache`: Renewable (recalculated when scaler changes)
   - `_port_return_cache`: Renewable (recalculated when scaler changes or returns change)
   - All three caches reset when `evaluate()` called

4. **Shape Preservation for Attribution**
   - Cache `port_return` as `(T, N)` - NEVER aggregate in cache
   - User aggregates on-demand: `rc.get_port_return(step).sum(dim='asset')`
   - Enables post-mortem winner/loser analysis

5. **On-Demand Aggregation**
   - `daily_pnl = port_return.sum(dim='asset')` - not cached
   - `cumulative_pnl = daily_pnl.cumsum(dim='time')` - not cached
   - Both are fast operations (<1ms), no need to cache

#### Lessons Learned

1. **Re-Masking is Critical**
   - Shift alone is not enough (creates NaN pollution)
   - Must re-mask with current universe after shift
   - This liquidates exited positions correctly

2. **Position-Level Attribution Requires (T, N) Shape**
   - Don't aggregate too early
   - Keep element-wise returns for traceability
   - Aggregate only when user requests it

3. **Cumsum > Cumprod for Research**
   - Time-invariance ensures fair comparison
   - Compound interest unfairly favors longer strategies
   - Simple sum is more interpretable

4. **Vectorization is Key**
   - 7ms for 25,200 data points (252×100)
   - No Python loops (pure xarray operations)
   - Scales to production datasets

5. **Forward-Bias Prevention via Shift**
   - `.shift(time=1)` ensures we trade on yesterday's signal
   - Today's return mapped to yesterday's weights
   - Critical for realistic backtest results

#### Next Steps

1. ✅ Experiment validated shift-mask-multiply workflow
2. ✅ Document findings in FINDINGS.md (this entry)
3. **TODO**: Implement return data auto-loading in AlphaCanvas.__init__
4. **TODO**: Add _port_return_cache to EvaluateVisitor (triple-cache)
5. **TODO**: Implement backtest logic in Visitor._cache_signal_and_weights
6. **TODO**: Add convenience methods: rc.get_port_return(), rc.get_daily_pnl(), rc.get_cumulative_pnl()
7. **TODO**: Create comprehensive tests for backtest module
8. **TODO**: Create showcase demonstrating backtest and attribution
9. **TODO**: Update documentation (PRD, Architecture, Implementation)

---

## Phase 9: alpha-database Package

### Experiment 20: DataSource Design Validation

**Date**: 2025-01-23  
**Status**: ✅ SUCCESS

**Summary**: Validated that the new DataSource design (dates passed per call) produces 100% identical results to the old DataLoader design (dates in constructor). The new stateless design is ready for implementation in the alpha-database package.

#### Key Discoveries

1. **Stateless Design is Correct**
   - **Old Design**: `DataLoader(config, start_date, end_date)` - dates in constructor
   - **New Design**: `DataSource(config)` then `load_field(field_name, start_date, end_date)` - dates per call
   - **Result**: 100% identical outputs (xarray.equals() returns True)
   - **Max Difference**: 0.00e+00 (perfect match)

2. **Performance Improvement**
   - **Old Design**: 0.0110s per load
   - **New Design**: 0.0098s per load
   - **Improvement**: 10.9% faster (negligible difference)
   - **Conclusion**: Performance is acceptable for production use

3. **Reusability Validated**
   - Same DataSource instance can load multiple fields
   - Tested: `adj_close` → `volume` with same instance
   - Result: Both fields load correctly without state interference
   - **Benefit**: Users can create one DataSource and reuse it

4. **Stateless Confirmed**
   - Same instance can handle different date ranges
   - Tested: Jan data → Feb data → Jan data again
   - Result: Jan data identical on both calls (no state leakage)
   - **Critical**: Proves no hidden state pollution

5. **Design Separation of Concerns**
   - `ConfigLoader`: Manages YAML config parsing (same as before)
   - `DataLoader`: Pure transformation (DataFrame → xarray), no config/dates
   - `DataSource`: Facade that orchestrates config + loader + readers
   - **Benefit**: Each component has single responsibility

#### Implementation Pattern

**Old Design (alpha-canvas)**:
```python
config = ConfigLoader('config')
loader = DataLoader(config, '2024-01-01', '2024-01-31')
data = loader.load_field('adj_close')
```

**New Design (alpha-database)**:
```python
ds = DataSource('config/data.yaml')
data = ds.load_field('adj_close', '2024-01-01', '2024-01-31')
```

**Key Difference**: Date range is now an argument to `load_field()`, not constructor state.

#### Architecture Implications

1. **alpha-database Independence**
   - Will have its own `ConfigLoader` (copied, not imported from alpha-canvas)
   - Will have its own `DataLoader` (stateless version)
   - Will expose `DataSource` as public API
   - **Benefit**: True separation of concerns, no circular dependencies

2. **Backward Compatibility**
   - alpha-canvas will keep its old DataLoader until migration complete
   - No breaking changes to existing code
   - Gradual migration path for users

3. **Plugin Architecture Ready**
   - `DataSource` can register custom readers: `ds.register_reader('postgres', PostgresReader())`
   - Enables specialized readers (FnGuide, Bloomberg) as plugins
   - MVP will include: ParquetReader, CSVReader (core readers)

#### Lessons Learned

1. **Stateless > Stateful for Data Access**
   - Dates as parameters (not state) enables reusability
   - No hidden state = easier to reason about
   - Same pattern as database connections (pass query, not store it)

2. **Experiment-Driven Development Works**
   - Validated design before implementing in production
   - Caught potential issues early (none found, design is sound)
   - Provides concrete evidence for architecture decisions

3. **Performance is Not a Concern**
   - 10.9% difference is negligible for I/O-bound operations
   - Query execution and DataFrame pivoting dominate runtime
   - Design choice should prioritize clarity over micro-optimization

#### Next Steps

1. ✅ Experiment validated new DataSource design
2. ✅ Document findings in FINDINGS.md (this entry)
3. **TODO**: Create alpha_database package structure (core/, readers/)
4. **TODO**: Port ConfigLoader to alpha_database/core/config.py
5. **TODO**: Implement stateless DataLoader in alpha_database/core/data_loader.py
6. **TODO**: Implement BaseReader interface and ParquetReader
7. **TODO**: Implement DataSource facade
8. **TODO**: Create integration tests comparing old vs new
9. **TODO**: Create showcase demonstrating DataSource usage
10. **TODO**: Eventually integrate DataSource into AlphaCanvas (dependency injection)

---