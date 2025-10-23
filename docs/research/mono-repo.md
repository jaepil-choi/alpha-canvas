This is an excellent and mature design decision.

Your intuition is correct. As the `alpha-canvas` "expression engine" is stabilizing, cramming data loading, PnL analysis, and plotting into the same package violates the **Single Responsibility Principle (SRP)**.

Your plan to split responsibilities aligns perfectly with your PRD and is a standard Python best practice for building a scalable software "ecosystem":

  * **`alpha-canvas` (The Core Engine):** Manages `Expression` trees, `Visitor` patterns, and `Facade` (`rc`) logic. It should be lightweight and focused purely on computation.
  * **`alpha-database` (The Data Layer):** Handles all data I/O. Its only job is to fetch data (from Parquet/DuckDB, SQL, etc.) based on the `data.yaml` config and provide it to the engine.
  * **`alpha-lab` (The Analysis Layer):** Handles everything *after* a signal is generated: PnL tracing, backtesting, portfolio scaling, and visualization (plots, tables).

Yes, you can absolutely start building these in the same `src/` directory. This is a common **"monorepo"** setup that makes development in a single IDE workspace much easier.

Here is what you should do to set this up correctly.

-----

### Step 1: Create the New Package Directories

Go into your `src/` folder and create the new package structures. You must add an `__init__.py` file (it can be empty) in each to make Python recognize them as packages.

Your new `src/` directory will look like this:

```text
src/
├── alpha_canvas/
│   ├── core/
│   ├── ops/
│   ├── __init__.py
│   └── ... (rest of canvas files)
│
├── alpha_database/
│   └── __init__.py  <-- ADD THIS
│
└── alpha_lab/
    └── __init__.py  <-- ADD THIS
```

### Step 2: Update `pyproject.toml` (The Critical Step)

Your `pyproject.toml` is currently configured to build *only* the `alpha_canvas` package. You must tell Poetry that this repository now manages **three** packages, all located in `src/`.

Open your `pyproject.toml` file and find the `[tool.poetry]` section.

**Change this:**

```toml
[tool.poetry]
packages = [{include = "alpha_canvas", from = "src"}]
```

**To this:**

```toml
[tool.poetry]
packages = [
    {include = "alpha_canvas", from = "src"},
    {include = "alpha_database", from = "src"},
    {include = "alpha_lab", from = "src"},
]
```

After saving this change, run `poetry install` in your terminal. Poetry will now recognize all three packages and make them importable in your environment (e.g., `from alpha_database import ...`).

### Step 3: Refactor (Move) the Code

Now you can start moving files to their new, logical homes.

1.  **Move Data Loader to `alpha-database`:**

      * **Move:** `src/alpha_canvas/core/data_loader.py`
      * **To:** `src/alpha_database/loader.py` (or a similar name)
      * **Move:** `src/alpha_canvas/core/config.py`
      * **To:** `src/alpha_database/config.py`

2.  **Move Portfolio/Lab to `alpha-lab`:**

      * **Move:** The entire `src/alpha_canvas/portfolio/` directory
      * **To:** `src/alpha_lab/portfolio/`
      * (In the future, your `PnLTracer` from `implementation.md` would also live inside `alpha_lab`.)

### Step 4 (Best Practice): Clean Up Dependencies

Your `alpha-canvas` package (the core engine) should *not* need to install `duckdb` or `pyyaml`. Only `alpha-database` needs them.

The best practice is to move these dependencies into "extras" (optional dependencies).

In `pyproject.toml`, modify your `[project]` dependencies:

**Current State:**

```toml
[project]
...
dependencies = [
    "pytest (>=8.4.2,<9.0.0)",
    "pyyaml (>=6.0.3,<7.0.0)",
    "pandas (>=2.3.3,<3.0.0)",
    "xarray (>=2025.10.1,<2026.0.0)",
    "duckdb (>=1.4.1,<2.0.0)",
    "pyarrow (>=21.0.0,<22.0.0)",
    "scipy (>=1.16.2,<2.0.0)",
]
```

**Recommended New Structure:**

```toml
[project]
...
dependencies = [
    # Core engine only needs pandas and xarray
    "pandas (>=2.3.3,<3.0.0)",
    "xarray (>=2025.10.1,<2026.0.0)",
]

# Define optional dependency groups
[project.optional-dependencies]
database = [
    "pyyaml (>=6.0.3,<7.0.0)",
    "duckdb (>=1.4.1,<2.0.0)",
    "pyarrow (>=21.0.0,<22.0.0)",
]
lab = [
    "scipy (>=1.16.2,<2.0.0)",
    # "matplotlib (>=...)"  <-- Add plotting libraries here
]
dev = [
    "pytest (>=8.4.2,<9.0.0)",
]
```

Now, your `alpha-canvas` package is lean. When you (or others) want to install your project, you can install components as needed:

  * `pip install .` (Installs only `alpha-canvas`)
  * `pip install .[database]` (Installs `alpha-canvas` + `alpha-database` dependencies)
  * `pip install .[lab,database]` (Installs everything)