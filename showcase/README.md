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

## Expected Output

Each showcase produces detailed terminal output showing:
- What operation is being performed
- Results and validation
- Success/failure indicators
- Performance characteristics

All showcases should complete with `[SUCCESS]` verdicts, demonstrating that the MVP foundation is solid and ready for use.

## Foundation Test Results

The complete MVP foundation has:
- ✅ **53 tests passing** (100% success rate)
- ✅ **8 experiments validated** (all SUCCESS)
- ✅ **5 phases complete** (Config, DataPanel, Expression, Facade, Parquet Data Loading)
- ✅ **Production-ready** architecture (Facade, Composite, Visitor patterns)
- ✅ **Data pipeline**: Parquet → DuckDB → xarray → AlphaCanvas (~5.38ms per field)

---

**Built with Experiment-Driven Development** 🔬

