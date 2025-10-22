# Alpha Canvas MVP - Showcase

This directory contains demonstration scripts that showcase the functionality of the alpha-canvas MVP foundation.

## Running the Showcases

Each showcase can be run independently:

```bash
poetry run python showcase/01_config_module.py
poetry run python showcase/02_datapanel_model.py
poetry run python showcase/03_expression_visitor.py
poetry run python showcase/04_facade_complete.py
poetry run python showcase/05_parquet_data_loading.py
poetry run python showcase/06_ts_mean_operator.py
poetry run python showcase/07_ts_any_surge_detection.py
poetry run python showcase/08_rank_market_cap.py
poetry run python showcase/09_universe_masking.py
poetry run python showcase/10_boolean_expressions.py
poetry run python showcase/11_data_accessor.py
poetry run python showcase/12_cs_quantile.py
poetry run python showcase/13_signal_assignment.py
poetry run python showcase/14_weight_scaling.py
poetry run python showcase/15_weight_caching_pnl.py
poetry run python showcase/16_backtest_attribution.py
```

Or run all showcases in sequence:

```bash
poetry run python showcase/run_all.py
```

## What Each Showcase Demonstrates

### 01. Config Module (`01_config_module.py`)
- Loading YAML configuration files
- Accessing field definitions
- Listing available fields
- Structure validation

**Key Features**: ConfigLoader, YAML parsing, field access

### 02. DataPanel Model (`02_datapanel_model.py`)
- Creating (T, N) panel data with xarray.Dataset
- Adding heterogeneous data types (float, string, bool)
- Eject pattern: Getting pure xarray.Dataset
- Inject pattern: Adding external DataArrays
- Boolean indexing for selection

**Key Features**: DataPanel wrapper, Open Toolkit (eject/inject), xarray integration

### 03. Expression & Visitor (`03_expression_visitor.py`)
- Creating Expression trees (Composite pattern)
- Evaluating expressions with Visitor pattern
- Integer-based step caching
- Depth-first traversal
- Intermediate result caching

**Key Features**: Expression ABC, Field nodes, EvaluateVisitor, step indexing

### 04. Complete Facade (`04_facade_complete.py`)
- Full AlphaCanvas integration
- Config + DataPanel + Expression + Visitor working together
- add_data() with both DataArray and Expression
- End-to-end workflow demonstration
- Real-world usage patterns

**Key Features**: AlphaCanvas facade, complete integration, production-ready interface

### 05. Parquet Data Loading (`05_parquet_data_loading.py`)
- Mock Parquet file inspection (long format)
- DuckDB SQL queries on Parquet files
- Date range filtering with parameter substitution
- Long-to-wide pivot transformation
- xarray.DataArray creation with coordinates
- AlphaCanvas lazy panel initialization
- Boolean mask filtering (time-series and cross-sectional)
- Computing with multiple fields (e.g., dollar volume)
- Adding computed results back to dataset

**Key Features**: DataLoader, DuckDB integration, data filtering, field operations, complete pipeline (~5.38ms per field)

### 06. Time-Series Operator - ts_mean() (`06_ts_mean_operator.py`)
- Manual Expression creation (TsMean)
- Helper function usage (rc.ts_mean())
- Rolling mean calculation with NaN padding
- Integer-based step caching (depth-first traversal)
- Cross-sectional independence verification
- Nested expressions (MA of MA)
- Edge cases (window=1, window>T)
- Visual comparison of multiple moving averages

**Key Features**: TsMean operator, polymorphic time-series design, Expression/Visitor pattern, step caching

### 07. Time-Series Operator - ts_any() (`07_ts_any_surge_detection.py`)
- Event detection with rolling boolean windows
- Rolling sum > 0 pattern for "any True in window"
- Surge detection (returns > threshold)
- Cross-sectional independence
- Window persistence verification
- Performance benchmarking
- Real-world event tracking scenarios

**Key Features**: TsAny operator, boolean window operations, event detection patterns, rolling.sum() optimization

### 08. Cross-Sectional Operator - rank() (`08_rank_market_cap.py`)
- Percentile ranking across assets (cross-sectional)
- Ascending rank (0.0 = smallest, 1.0 = largest)
- NaN handling with scipy.stats.rankdata
- Time independence verification
- Integration with Field and Visitor
- Market cap ranking example
- Multiple time steps analysis

**Key Features**: Rank operator, scipy integration, cross-sectional operations, ordinal ranking, percentile conversion

### 09. Universe Masking (`09_universe_masking.py`)
- Investable universe definition
- Automatic universe masking (double masking strategy)
- Input masking at Field retrieval
- Output masking at operator results
- Idempotent masking verification
- Time-varying universe (delisting scenario)
- Open Toolkit integration (injected data also masked)
- Universe coverage statistics
- Edge cases (penny stocks, illiquid stocks)

**Key Features**: Universe masking, double masking trust chain, immutable universe, xarray.where() pattern, realistic penny stock scenario

### 10. Boolean Expressions (`10_boolean_expressions.py`)
- Comparison operators creating Expressions (==, !=, <, >, <=, >=)
- Logical operators combining Expressions (&, |, ~)
- Lazy evaluation (Expressions created, not evaluated)
- Evaluation through EvaluateVisitor (Visitor-based, not standalone)
- Universe masking applied automatically
- String and numeric comparisons
- Chained boolean logic
- Visitor step caching for all Expression types

**Key Features**: Boolean Expression operators, lazy comparison, Visitor pattern integration, Expression-based selector foundation

### 11. DataAccessor - Selector Interface (`11_data_accessor.py`)
- `rc.data` accessor returning Field Expressions
- Lazy evaluation: `rc.data['size'] == 'small'` creates Expression
- Complex logical chains with `&`, `|`, `~` operators
- Multi-dimensional selection (Fama-French style)
- Numeric and categorical comparisons
- Evaluation with universe masking
- Pythonic syntax without special "axis" accessor

**Key Features**: DataAccessor, Expression-based selectors, lazy field access, integrated Phase 7A+7B, simplified design

### 12. cs_quantile - Cross-Sectional Quantile Bucketing (`12_cs_quantile.py`)
- Independent sort: Quantile bucketing across entire universe
- Dependent sort: Quantile bucketing within groups (Fama-French methodology)
- Shape preservation: `(T, N)` input → `(T, N)` categorical output
- Categorical labels: 'small'/'big', 'low'/'mid'/'high'
- Integration with Boolean Expressions: `rc.data['size'] == 'small'`
- Multi-dimensional portfolio construction (2×3 Fama-French portfolios)
- Different cutoffs per group (validates correct dependent sort behavior)

**Key Features**: cs_quantile operator, independent vs dependent sort, Fama-French portfolios, quantile bucketing, xarray groupby with shape preservation

### 13. Signal Canvas & Lazy Assignment (`13_signal_assignment.py`)
- Implicit blank canvas (Constant Expression with value 0.0)
- Lazy signal assignment: `signal[mask] = value` stores without evaluating
- Fama-French 2×3 factor construction (Size × Value)
- Boolean selector masks with logical operators (`&`, `|`)
- Sequential assignment application (later assignments overwrite earlier ones)
- Traceability: Base result and final result cached separately
- Overlapping mask handling demonstration
- Actual data verification (no hardcoded results)
- Complete end-to-end signal construction workflow

**Key Features**: Lazy assignment, Constant Expression, implicit canvas, sequential application semantics, Visitor integration, traceability, Fama-French factor construction

### 14. Portfolio Weight Scaling - Strategy Pattern (`14_weight_scaling.py`)
- Strategy Pattern for weight scaling (easy to swap and extend)
- GrossNetScaler: Unified gross/net exposure framework (vectorized)
- DollarNeutralScaler: Convenience wrapper (L=1.0, S=-1.0)
- One-sided signal handling (gross target always met)
- NaN preservation (universe integration)
- Comparison of different scaling strategies
- Default parameters: `target_gross=2.0`, `target_net=0.0`
- Performance: 7-40ms for typical datasets
- Explicit scaler selection (no default scaler)
- Easy extensibility: SoftmaxScaler, RiskTargetScaler, etc.

**Key Features**: Strategy Pattern, GrossNetScaler vectorized algorithm, dollar-neutral portfolio, net-long bias, stateless design, explicit strategy selection, performance validated

### 15. Weight Caching & PnL Tracing (`15_weight_caching_pnl.py`)
- Dual-cache architecture (signals + weights) for PnL analysis
- Step-by-step weight tracking at every evaluation step
- Efficient scaler replacement without re-evaluation
- `rc.evaluate(expr, scaler=...)` API for weight caching
- `rc.get_weights(step)` convenience method for weight retrieval
- Signal cache persistence, weight cache renewal
- Preparing for future PnL tracing and attribution
- Multi-step expression example (Field → TsMean → Rank)
- Comparison of DollarNeutral vs. Net-Long strategies
- Future API preview for step-by-step PnL decomposition

**Key Features**: Dual-cache architecture, step-by-step weight caching, efficient scaler swapping, PnL tracing foundation, rc.get_weights() API, signal persistence, weight renewal

### 16. Backtest & Attribution (`16_backtest_attribution.py`)
- Automatic backtest execution when scaler provided
- Position-level returns (T, N) for winner/loser attribution
- On-demand daily and cumulative PnL aggregation
- Shift-mask workflow preventing forward-looking bias
- Re-masking preventing NaN pollution from universe exits
- Efficient scaler comparison (signal cache reused)
- Step-by-step PnL decomposition
- Cumsum-based fair strategy comparison

**Key Features**: Triple-cache architecture (signal + weight + port_return), position-level attribution, shift-mask workflow, forward-bias prevention, winner/loser analysis, rc.get_port_return(), rc.get_daily_pnl(), rc.get_cumulative_pnl()

## Expected Output

Each showcase produces detailed terminal output showing:
- What operation is being performed
- Results and validation
- Success/failure indicators
- Performance characteristics

All showcases should complete with `[SUCCESS]` verdicts, demonstrating that the MVP foundation is solid and ready for use.

## Foundation Test Results

The complete MVP foundation has:
- ✅ **176 tests passing** (100% success rate)
- ✅ **19 experiments validated** (all SUCCESS)
- ✅ **17 phases complete**: Config, DataPanel, Expression, Facade, Parquet Data Loading, TsMean, Refactoring, TsAny, Rank, Universe Masking, Boolean Expressions, DataAccessor, cs_quantile, Signal Assignment, Weight Scaling, Weight Caching, **Backtesting with Portfolio Returns**
- ✅ **Production-ready** architecture (Facade, Composite, Visitor, Strategy patterns)
- ✅ **Data pipeline**: Parquet → DuckDB → xarray → AlphaCanvas (~5.38ms per field)
- ✅ **Operator library**: TsMean, TsAny, Rank, CsQuantile, Constant, Equals, GreaterThan, And, Or, Not (+ 6 more logical operators)
- ✅ **Universe masking**: Automatic double-masking with 13.6% overhead
- ✅ **Boolean Expressions**: Lazy evaluation with Expression-based comparisons
- ✅ **Selector Interface**: `rc.data` accessor with Field Expressions (Phase 7B)
- ✅ **Quantile Bucketing**: Independent and dependent sort for Fama-French portfolios
- ✅ **Signal Assignment**: Lazy assignment with implicit canvas and traceability (Phase 7D)
- ✅ **Weight Scaling**: Strategy Pattern with GrossNetScaler (vectorized, 7-40ms, target_gross=2.0, target_net=0.0)
- ✅ **Weight Caching**: Dual-cache architecture (signals + weights) for step-by-step PnL tracing and efficient scaler comparison
- ✅ **Backtesting**: Triple-cache with position-level returns, shift-mask workflow, forward-bias prevention, winner/loser attribution (7ms for 252×100)

---

**Built with Experiment-Driven Development** 🔬

