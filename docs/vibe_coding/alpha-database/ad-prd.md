# Alpha-Database 제품 요구사항 문서 (Product Requirement Document)

## 1. 개요

**alpha-database**는 퀀트 리서치를 위한 **통합 데이터 인프라 패키지**입니다. 단순한 저장소를 넘어, 데이터 수집(fetching), 저장(storage), 검색(retrieval), 그리고 계보 추적(lineage tracking)까지 담당하는 엔드-투-엔드 데이터 관리 솔루션입니다.

### 역할 및 책임

1. **데이터 수집 (Data Fetching)**: 외부 API에서 데이터 다운로드 및 내부 저장
2. **데이터 검색 (Data Retrieval)**: Config-driven 데이터 로딩 (alpha-canvas에서 포팅)
3. **데이터 저장 (Data Storage)**: Alpha, factor, dataset 영구 저장
4. **메타데이터 관리 (Metadata)**: 데이터 카탈로그 및 계보 추적

### 핵심 원칙

* **Loosely Coupled**: alpha-canvas 공개 API를 통해 데이터 추출/제공
* **Source Agnostic**: 다양한 데이터 소스 지원 (Parquet, CSV, Excel, API, DB)
* **Format Aware**: 소스별 특수 포맷 처리 (FnGuide, Bloomberg 등)
* **Lineage Tracking**: 데이터 의존성 그래프 유지
* **Reproducibility**: Expression 직렬화로 재현성 보장

### 패키지 구조

```python
alpha_database/
├── __init__.py
├── core/
│   ├── config.py          # ConfigLoader (포팅: alpha-canvas)
│   ├── data_loader.py     # DataLoader (포팅: alpha-canvas)
│   └── data_source.py     # 통합 DataSource facade
├── readers/               # 소스별 데이터 리더
│   ├── base.py            # BaseReader ABC
│   ├── parquet.py         # ParquetReader
│   ├── csv.py             # CSVReader
│   ├── excel.py           # ExcelReader (generic)
│   ├── fnguide.py         # FnGuideExcelReader (special format)
│   ├── bloomberg.py       # BloombergReader (future)
│   └── postgres.py        # PostgresReader (future)
├── fetchers/              # 데이터 수집 (NEW)
│   ├── base.py            # BaseFetcher ABC
│   ├── ccxt_fetcher.py    # CCXT crypto data fetcher
│   ├── api_fetcher.py     # Generic API fetcher
│   └── scheduler.py       # Fetch scheduling & orchestration
├── writers/               # 데이터 저장
│   ├── dataset.py         # DatasetWriter (computed fields)
│   ├── alpha.py           # AlphaWriter (versioned alphas)
│   └── factor.py          # FactorWriter (factor returns)
├── catalog/               # 메타데이터 관리
│   ├── meta_table.py      # 중앙 메타 테이블
│   ├── lineage.py         # 의존성 그래프 관리
│   └── explorer.py        # 카탈로그 탐색 유틸리티
├── serialization/         # Expression 직렬화 (NEW)
│   ├── expression.py      # Expression.to_dict() / from_dict()
│   └── cache.py           # Evaluator step cache 직렬화
└── integration/           # alpha-canvas 통합
    └── canvas_adapter.py  # AlphaCanvas 어댑터
```

## 2. 우선순위 및 구현 로드맵

### **P0 (MVP - Essential)**: 데이터 검색 (Data Retrieval)
1. Config-driven data loading (포팅: alpha-canvas)
2. Multi-source reader facade
3. Alpha-canvas 통합 및 검증

### **P1 (Core)**: 데이터 저장 (Data Storage)
1. Dataset catalog (computed fields)
2. Alpha catalog (versioned alphas with Expression)
3. Factor catalog (factor returns with Expression)
4. 중앙 메타 테이블

### **P2 (Important)**: 데이터 수집 (Data Fetching)
1. CCXT crypto fetcher
2. API fetcher framework
3. Hive-partitioned Parquet storage
4. Scheduling & error handling

### **P3 (Enhancement)**: 계보 추적 (Lineage Tracking)
1. 의존성 그래프 관리
2. Impact analysis tools

### **P4 (Future)**: 고급 기능
1. PostgreSQL backend
2. ClickHouse backend
3. Data quality checks

---

## 3. P0: Config-Driven Data Loading (MVP)

### 개요

alpha-canvas의 `core/config.py`와 `core/data_loader.py`를 alpha-database로 포팅합니다. 이는 alpha-database의 **핵심 가치 제안**이며, alpha-canvas와의 통합 지점입니다.

### 요구사항

**R1**: 기존 `config/data.yaml` 파싱 및 필드 정의 로드
**R2**: Parameterized SQL 쿼리 지원 (`:start_date`, `:end_date`)
**R3**: Long → Wide pivoting (date × security_id → time × asset)
**R4**: `xarray.DataArray` 반환 with `(time, asset)` dimensions
**R5**: Backward compatibility: 기존 alpha-canvas 코드 변경 최소화

### 포팅 범위

**From alpha-canvas**:
- `src/alpha_canvas/core/config.py` → `alpha_database/core/config.py`
- `src/alpha_canvas/core/data_loader.py` → `alpha_database/core/data_loader.py`

**새로 구현**:
- `alpha_database/core/data_source.py`: 통합 facade

### 사용 시나리오

```python
from alpha_database import DataSource

# 시나리오 1: 기존 config 사용
ds = DataSource(config_path='config/data.yaml')
data = ds.load_field(
    'adj_close',
    start_date='2024-01-01',
    end_date='2024-12-31'
)
# Returns: xr.DataArray with (time, asset) dimensions

# 시나리오 2: Alpha-canvas 통합
from alpha_canvas import AlphaCanvas

rc = AlphaCanvas(data_source=ds)  # Dependency injection
rc.add_data('price', Field('adj_close'))  # Triggers ds.load_field()
```

---

## 4. P0: Multi-Source Reader Facade

### 개요

다양한 데이터 소스(Parquet, CSV, Excel, DB)와 특수 포맷(FnGuide, Bloomberg)을 통합 인터페이스로 지원합니다.

### 설계 원칙

**원칙 1**: **독립적 Reader 구현** (not layered abstraction)
- `ExcelReader`: Generic Excel 파일 (*.xlsx, *.xls)
- `FnGuideExcelReader`: FnGuide 특수 포맷 Excel
- 각 Reader는 독립적으로 구현, 일부 코드 중복 허용

**원칙 2**: **Source Agnostic** facade
- 사용자는 Reader 선택 불필요
- Config에 `db_type` 또는 `reader` 지정 시 자동 선택

### Reader 구조

```python
# Base interface
class BaseReader(ABC):
    @abstractmethod
    def read(
        self,
        query: str,
        params: dict
    ) -> pd.DataFrame:
        """Read data and return long-format DataFrame."""
        pass
```

**구현 목록 (MVP)**:
1. **ParquetReader**: DuckDB SQL on Parquet files (현재 구현 포팅)
2. **CSVReader**: Generic CSV files
3. **ExcelReader**: Generic Excel files
4. **FnGuideExcelReader**: FnGuide-specific Excel format

**Future**:
5. **BloombergReader**: Bloomberg CSV/API
6. **PostgresReader**: PostgreSQL database

### Config 예시

```yaml
# config/data.yaml

# Parquet source
adj_close:
  reader: parquet  # or db_type: parquet
  query: >
    SELECT date, security_id, close * adj_factor as adj_close
    FROM read_parquet('data/pricevolume.parquet')
    WHERE date >= :start_date AND date <= :end_date

# FnGuide Excel source (special format)
fnguide_financials:
  reader: fnguide_excel
  file_path: 'data/fnguide/financials_2024.xlsx'
  sheet_name: 'IS'
  header_row: 3  # FnGuide-specific: skip header rows
  date_col: '결산일'
  security_col: '종목코드'
```

### 통합 사용

```python
from alpha_database import DataSource

ds = DataSource(config_path='config/data.yaml')

# Automatically selects ParquetReader
data1 = ds.load_field('adj_close', start_date='2024-01-01', end_date='2024-12-31')

# Automatically selects FnGuideExcelReader
data2 = ds.load_field('fnguide_financials')
```

---

## 5. P0: Alpha-Canvas 통합

### 개요

Alpha-canvas가 alpha-database를 **의존성 주입(Dependency Injection)** 패턴으로 사용하도록 통합합니다.

### 통합 패턴

```python
from alpha_canvas import AlphaCanvas
from alpha_database import DataSource

# Step 1: DataSource 초기화
ds = DataSource(config_path='config/data.yaml')

# Step 2: Alpha-canvas에 주입
rc = AlphaCanvas(
    start_date='2024-01-01',
    end_date='2024-12-31',
    data_source=ds  # Dependency injection
)

# Step 3: Field 사용 시 자동으로 DataSource 호출
rc.add_data('price', Field('adj_close'))
# Internally: ds.load_field('adj_close', start_date=..., end_date=...)
```

### Backward Compatibility 전략

**Phase 1**: Dual mode (alpha-canvas 유지)
```python
# Option A: 기존 방식 (backward compatible)
rc = AlphaCanvas(start_date='2024-01-01', end_date='2024-12-31')
# 내부적으로 기존 data_loader 사용

# Option B: 새로운 방식 (alpha-database)
ds = DataSource(config_path='config/data.yaml')
rc = AlphaCanvas(start_date='2024-01-01', end_date='2024-12-31', data_source=ds)
# alpha-database 사용
```

**Phase 2**: 모든 테스트를 alpha-database로 전환

**Phase 3**: 기존 `core/config.py`, `core/data_loader.py` 제거

---

## 6. P1: Dataset Catalog (Computed Field Storage)

### 개요

계산된 필드(derived characteristics)를 저장하고 스키마를 동적으로 진화시킵니다. 예: PBR, EV/EBITDA 등의 파생 지표를 계산하여 재사용합니다.

### 요구사항

**R1**: Dataset 생성 및 컬럼 추가 (schema evolution)  
**R2**: 의존성 추적 (어떤 raw field에서 파생되었는지)  
**R3**: Upsert 로직 (날짜 중복 시 덮어쓰기)  
**R4**: 중앙 메타 테이블 자동 업데이트

### 사용 시나리오

```python
from alpha_database import DatasetWriter

writer = DatasetWriter(base_path='./data')

# 시나리오 1: PBR 계산 및 저장
pbr = price / book_value
result = rc.evaluate(pbr)

writer.save_field(
    dataset_name='fundamental',
    field_name='pbr',
    data=result,  # xr.DataArray (T, N)
    dependencies=['adj_close', 'book_value'],  # Lineage tracking
    metadata={'description': 'Price to Book Ratio', 'created': '2025-01-23'}
)
# Result: data/fundamental.parquet with [date, asset, pbr]
# Meta table updated automatically

# 시나리오 2: 동일 dataset에 EV/EBITDA 추가 (schema evolution)
ev_ebitda = ev / ebitda
result2 = rc.evaluate(ev_ebitda)

writer.save_field(
    dataset_name='fundamental',  # SAME dataset
    field_name='ev_ebitda',      # NEW column
    data=result2,
    dependencies=['ev', 'ebitda']
)
# Result: fundamental.parquet now [date, asset, pbr, ev_ebitda]

# 시나리오 3: 나중에 로드
ds = DataSource(config_path='config/data.yaml')
fundamental = ds.load_field('pbr')  # From fundamental.parquet
```

---

## 7. P1: Alpha Catalog (Versioned Alpha with Expression)

### 개요

완전한 알파(signal + weights + returns + Expression)를 메타데이터와 함께 저장합니다. **Expression 직렬화**를 통해 재현성을 보장합니다.

### 핵심 요구사항 (Critical)

**R1**: **자동 데이터 추출** - `canvas`와 `step`만 전달, 자동으로 signal/weights/returns 추출  
**R2**: **Expression 직렬화** - Expression tree를 구조화된 dict로 저장 (재현 가능)  
**R3**: **인간 가독성** - `str(expr)`도 함께 저장 (디버깅용)  
**R4**: **의존성 추적** - Expression에서 자동으로 의존 field 추출  
**R5**: **버전 관리** - alpha_id로 자동 버전 관리 (v1, v2, v3)  
**R6**: **No 성과 지표** - Sharpe 등 저장하지 않음 (alpha-lab에서 on-demand 계산)

### Evaluator 수정 필요사항

**현재**: Evaluator는 data (signal, weights, port_return)만 캐싱  
**필요**: Evaluator는 **Expression steps**도 캐싱해야 함

```python
# In EvaluateVisitor
class EvaluateVisitor:
    def __init__(self, ...):
        self._signal_cache: Dict[int, xr.DataArray] = {}
        self._weight_cache: Dict[int, Optional[xr.DataArray]] = {}
        self._port_return_cache: Dict[int, Optional[xr.DataArray]] = {}
        
        # NEW: Expression cache
        self._expression_cache: Dict[int, Expression] = {}  # Store Expression objects!
    
    def visit_operator(self, node):
        ...
        # Cache the Expression itself
        self._expression_cache[self._step_counter] = node
        ...
```

### Expression 직렬화

```python
# In alpha_database/serialization/expression.py

class ExpressionSerializer:
    @staticmethod
    def to_dict(expr: Expression) -> dict:
        """Serialize Expression to dict (reproducible)."""
        if isinstance(expr, Field):
            return {
                'type': 'Field',
                'name': expr.name
            }
        elif isinstance(expr, TsMean):
            return {
                'type': 'TsMean',
                'child': ExpressionSerializer.to_dict(expr.child),
                'window': expr.window
            }
        elif isinstance(expr, Rank):
            return {
                'type': 'Rank',
                'child': ExpressionSerializer.to_dict(expr.child),
                'ascending': expr.ascending
            }
        # ... handle all Expression types
    
    @staticmethod
    def from_dict(data: dict) -> Expression:
        """Deserialize dict to Expression (reconstruct)."""
        if data['type'] == 'Field':
            return Field(data['name'])
        elif data['type'] == 'TsMean':
            child = ExpressionSerializer.from_dict(data['child'])
            return TsMean(child, window=data['window'])
        # ... reconstruct all Expression types
    
    @staticmethod
    def extract_dependencies(expr: Expression) -> List[str]:
        """Extract all Field dependencies from Expression tree."""
        deps = []
        if isinstance(expr, Field):
            deps.append(expr.name)
        elif hasattr(expr, 'child'):
            deps.extend(ExpressionSerializer.extract_dependencies(expr.child))
        elif hasattr(expr, 'left') and hasattr(expr, 'right'):
            deps.extend(ExpressionSerializer.extract_dependencies(expr.left))
            deps.extend(ExpressionSerializer.extract_dependencies(expr.right))
        return list(set(deps))  # Deduplicate
```

### 사용 시나리오

```python
from alpha_database import AlphaWriter
from alpha_canvas import AlphaCanvas, Field
from alpha_canvas.ops import TsMean, Rank
from alpha_canvas.portfolio import DollarNeutralScaler

# Step 1: Create alpha in alpha-canvas
rc = AlphaCanvas(start_date='2024-01-01', end_date='2024-12-31')
expr = Rank(TsMean(Field('returns'), window=5))
result = rc.evaluate(expr, scaler=DollarNeutralScaler())

# Step 2: Save alpha with auto-extraction
writer = AlphaWriter(base_path='./alphas')

writer.save_alpha(
    alpha_id='momentum_ma5_rank',  # Will auto-version as v1, v2, etc.
    canvas=rc,                     # Pass canvas reference
    step=2,                        # Specify step (0=Field, 1=TsMean, 2=Rank)
    metadata={
        'description': 'Momentum strategy with 5-day MA and rank',
        'created': '2025-01-23',
        'author': 'researcher_name',
        'tags': ['momentum', 'mean-reversion']
    }
)

# Internally, AlphaWriter does:
# 1. Auto-extract: signal = rc.get_signal(step=2)
# 2. Auto-extract: weights = rc.get_weights(step=2)
# 3. Auto-extract: returns = rc.get_port_return(step=2)
# 4. Auto-extract: expr = rc._evaluator._expression_cache[2]  # NEW!
# 5. Serialize: expr_dict = ExpressionSerializer.to_dict(expr)
# 6. Dependencies: deps = ExpressionSerializer.extract_dependencies(expr)  # ['returns']
# 7. Save to disk:
#    - alphas/momentum_ma5_rank_v1/signal.parquet
#    - alphas/momentum_ma5_rank_v1/weights.parquet
#    - alphas/momentum_ma5_rank_v1/returns.parquet
#    - alphas/momentum_ma5_rank_v1/metadata.json:
#      {
#        'alpha_id': 'momentum_ma5_rank',
#        'version': 1,
#        'step': 2,
#        'expression': {  # Serialized Expression (machine-readable)
#          'type': 'Rank',
#          'child': {
#            'type': 'TsMean',
#            'child': {'type': 'Field', 'name': 'returns'},
#            'window': 5
#          },
#          'ascending': False
#        },
#        'expression_str': 'Rank(TsMean(Field("returns"), 5))',  # Human-readable
#        'dependencies': ['returns'],
#        'description': 'Momentum strategy with 5-day MA and rank',
#        'created': '2025-01-23',
#        'author': 'researcher_name',
#        'tags': ['momentum', 'mean-reversion']
#      }
# 8. Update meta_table:
#    name='momentum_ma5_rank_v1', type='alpha', location='alphas/momentum_ma5_rank_v1/', dependencies=['returns']
```

### 알파 재현 (Reproducibility)

```python
from alpha_database import AlphaReader

reader = AlphaReader(base_path='./alphas')

# Load alpha metadata
metadata = reader.load_alpha_metadata('momentum_ma5_rank_v1')

# Reconstruct Expression
expr_reconstructed = ExpressionSerializer.from_dict(metadata['expression'])

# Re-run with same Expression (exact reproduction)
rc_new = AlphaCanvas(start_date='2024-01-01', end_date='2024-12-31')
result_reproduced = rc_new.evaluate(expr_reconstructed, scaler=DollarNeutralScaler())

# Or load pre-computed data
alpha_data = reader.load_alpha_data('momentum_ma5_rank_v1')
# Returns: {'signal': DataArray, 'weights': DataArray, 'returns': DataArray, 'metadata': dict}
```

---

## 8. P1: Factor Catalog (Factor Returns with Expression)

### 개요

팩터 수익률 time series와 생성 Expression을 함께 저장합니다. Alpha와 유사하지만 **time series `(T,)` shape**로 저장됩니다.

### 요구사항

**R1**: 팩터 수익률 `(T,)` 저장  
**R2**: Expression 직렬화 (어떻게 생성되었는지)  
**R3**: 의존성 추적  
**R4**: 버전 관리 (선택적)

### 사용 시나리오

```python
from alpha_database import FactorWriter

writer = FactorWriter(base_path='./factors')

# Step 1: Generate factor in alpha-canvas
# ... (Fama-French SMB example)
expr = Constant(0.0)
expr[rc.data['size'] == 'small'] = 1.0
expr[rc.data['size'] == 'big'] = -1.0
result = rc.evaluate(expr, scaler=DollarNeutralScaler())

# Step 2: Save factor
writer.save_factor(
    factor_id='fama_french_smb',
    canvas=rc,
    step=final_step,
    metadata={
        'description': 'Fama-French Size Factor (Small Minus Big)',
        'construction': 'Independent 2x3 sort on size/value',
        'rebalance': 'monthly',
        'universe': 'KOSPI200'
    }
)

# Saved structure:
# - factors/fama_french_smb/daily_returns.parquet (T,) shape
# - factors/fama_french_smb/metadata.json (with Expression)
```

---

## 9. P1: Meta Table (Central Metadata Registry)

### 개요

모든 dataset, alpha, factor의 메타데이터를 **단일 테이블**로 관리합니다. 복잡한 ORM 대신 간단한 Parquet 테이블 사용.

### Schema

```python
meta_table = pd.DataFrame({
    'name': str,           # 'adj_close', 'pbr', 'momentum_v1', etc.
    'type': str,           # 'raw_field', 'derived_field', 'alpha', 'factor'
    'location': str,       # File path
    'dependencies': list,  # List of dependent field names
    'created': datetime,   # Creation timestamp
    'updated': datetime,   # Last update
    'description': str,    # Human-readable description
    'tags': list,          # Searchable tags
    'version': int         # For alphas/factors (optional)
})
```

### 사용 시나리오

```python
from alpha_database import MetaTable

meta = MetaTable(path='./data/meta_table.parquet')

# Query 1: List all datasets
datasets = meta.query(type='derived_field')

# Query 2: Find all entities dependent on 'adj_close'
dependents = meta.find_dependents('adj_close')
# Returns: ['pbr', 'momentum_v1', 'value_factor', ...]

# Query 3: Search by tags
momentum_alphas = meta.search(tags__contains='momentum')

# Query 4: Get lineage graph
lineage = meta.get_lineage('momentum_v1')
# Returns: momentum_v1 → pbr → [adj_close, book_value]
```

### 자동 업데이트

모든 Writer (DatasetWriter, AlphaWriter, FactorWriter)는 저장 시 자동으로 meta_table 업데이트:
```python
# When saving field
writer.save_field(..., dependencies=['adj_close', 'book_value'])
# → meta_table에 자동 추가

# When saving alpha
writer.save_alpha(..., canvas=rc, step=2)
# → Expression에서 dependencies 자동 추출 → meta_table 업데이트
```

---

## 10. P2: Data Fetching (Infrastructure Feature)

### 개요

외부 API에서 데이터를 다운로드하여 내부 **hive-partitioned Parquet** 파일로 저장합니다. 이는 인프라 수준의 기능이지만 alpha-database의 필수 기능입니다.

### 요구사항

**R1**: CCXT 통합 (crypto intraday data)  
**R2**: Generic API fetcher framework  
**R3**: Hive partitioning (`year=2025/month=01/day=23/`)  
**R4**: Scheduling & orchestration  
**R5**: Error handling & retry logic  
**R6**: Incremental updates (fetch only new data)

### CCXT Crypto Fetcher

```python
from alpha_database import CCXTFetcher

fetcher = CCXTFetcher(
    exchange='binance',
    base_path='./data/crypto'
)

# Scenario 1: Fetch single symbol
fetcher.fetch_ohlcv(
    symbol='BTC/USDT',
    timeframe='1h',
    start_date='2025-01-01',
    end_date='2025-01-23'
)
# Saves to: data/crypto/binance/BTCUSDT/year=2025/month=01/part-*.parquet

# Scenario 2: Fetch multiple symbols (parallel)
fetcher.fetch_multiple(
    symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    timeframe='1h',
    start_date='2025-01-01'
)

# Scenario 3: Incremental update (fetch only new data)
fetcher.update_latest(
    symbol='BTC/USDT',
    timeframe='1h'
)
# Automatically detects last saved timestamp, fetches only new data
```

### Generic API Fetcher

```python
from alpha_database import APIFetcher

# Scenario: Custom API (e.g., FnGuide API, KRX API)
fetcher = APIFetcher(
    api_key='your_api_key',
    base_url='https://api.fnguide.com/v1'
)

response = fetcher.fetch(
    endpoint='/financials',
    params={'ticker': '005930', 'quarter': '2024Q4'}
)

# Transform & save
df = fetcher.transform_response(response)
fetcher.save_to_parquet(
    df,
    base_path='./data/fnguide/financials',
    partition_cols=['year', 'quarter']
)
# Saves to: data/fnguide/financials/year=2024/quarter=Q4/part-*.parquet
```

### Scheduling (Future Enhancement)

```python
from alpha_database import FetchScheduler

scheduler = FetchScheduler()

# Schedule daily crypto data fetch
scheduler.add_job(
    job_id='crypto_hourly',
    fetcher=CCXTFetcher(exchange='binance'),
    schedule='0 * * * *',  # Every hour
    symbols=['BTC/USDT', 'ETH/USDT']
)

# Schedule weekly financial data fetch
scheduler.add_job(
    job_id='fnguide_weekly',
    fetcher=FnGuideFetcher(),
    schedule='0 0 * * 0',  # Every Sunday
    universe='KOSPI200'
)

scheduler.start()
```

### Hive Partitioning Rationale

**장점**:
- Parquet 파일을 날짜별로 분할하여 query 성능 향상
- DuckDB SQL로 효율적으로 필터링: `WHERE year=2025 AND month=01`
- 증분 업데이트 용이: 최신 파티션만 덮어쓰기

**구조 예시**:
```
data/crypto/binance/BTCUSDT/
├── year=2024/
│   ├── month=12/
│   │   ├── day=01/part-0.parquet
│   │   ├── day=02/part-0.parquet
│   │   └── ...
│   └── ...
└── year=2025/
    ├── month=01/
    │   ├── day=01/part-0.parquet
    │   ├── day=02/part-0.parquet
    │   └── ...
    └── ...
```

**DuckDB 쿼리**:
```sql
SELECT *
FROM read_parquet('data/crypto/binance/BTCUSDT/**/*.parquet',
                  hive_partitioning=true)
WHERE year = 2025 AND month = 1 AND day BETWEEN 10 AND 20
```

---

## 11. P3: Lineage Tracking (Dependency Graph)

### 개요

데이터 의존성을 **그래프 구조**로 추적합니다. Impact analysis 및 재현성 보장에 필수적입니다.

### 요구사항

**R1**: 의존성 그래프 생성 및 저장  
**R2**: Impact analysis (상위 의존 항목 찾기)  
**R3**: Lineage visualization (future)

### 그래프 구조

```
Raw Fields (Leaf nodes):
    - adj_close
    - book_value
    - ev
    - ebitda

Derived Fields:
    - pbr → depends on [adj_close, book_value]
    - ev_ebitda → depends on [ev, ebitda]

Alphas:
    - momentum_v1 → depends on [pbr]  (via Expression)
    - value_factor → depends on [pbr, ev_ebitda]

Factors:
    - fama_french_smb → depends on [adj_close, book_value]  (via size classification)
```

### Implementation (Simple Approach)

**Option A**: Store in meta_table as list column (already done above)

**Option B**: Separate graph table (if complex queries needed)
```python
dependency_graph = pd.DataFrame({
    'child': str,   # 'momentum_v1'
    'parent': str,  # 'pbr'
    'depth': int    # 1 (pbr → momentum_v1)
})
```

### 사용 시나리오

```python
from alpha_database import LineageTracker

tracker = LineageTracker(meta_table_path='./data/meta_table.parquet')

# Query 1: Find all dependents of 'adj_close'
dependents = tracker.find_dependents('adj_close')
# Returns: ['pbr', 'momentum_v1', 'value_factor', 'fama_french_smb', ...]

# Query 2: Find all dependencies of 'momentum_v1'
dependencies = tracker.find_dependencies('momentum_v1')
# Returns: ['pbr', 'adj_close', 'book_value']  (recursive)

# Query 3: Impact analysis
# If 'adj_close' data changes, what needs to be recomputed?
impact = tracker.get_impact('adj_close')
# Returns: ['pbr', 'momentum_v1', 'value_factor', ...]  (in topological order)

# Query 4: Get full lineage path
lineage = tracker.get_lineage('momentum_v1')
# Returns: momentum_v1 → pbr → [adj_close, book_value]
```

### Automatic Dependency Extraction

```python
# When saving alpha, dependencies are auto-extracted from Expression
writer.save_alpha(canvas=rc, step=2, ...)

# Internally:
expr = rc._evaluator._expression_cache[2]
deps = ExpressionSerializer.extract_dependencies(expr)  # ['returns', 'book_value', ...]
# → Save to meta_table
```

---

## 12. MVP Implementation Priorities

### Phase 1: Data Retrieval (P0) - 2-3 weeks
1. ✅ Port ConfigLoader from alpha-canvas
2. ✅ Port DataLoader from alpha-canvas
3. ✅ Implement DataSource facade
4. ✅ Implement ParquetReader
5. ✅ Implement CSVReader
6. ✅ Implement ExcelReader
7. ✅ Implement FnGuideExcelReader
8. ✅ Alpha-canvas integration (dependency injection)
9. ✅ Backward compatibility validation
10. ✅ All tests pass with alpha-database

### Phase 2: Data Storage (P1) - 2-3 weeks
1. ✅ Implement DatasetWriter
2. ✅ Implement Meta Table
3. ✅ Implement Expression serialization (to_dict/from_dict)
4. ✅ Modify Evaluator to cache Expression steps
5. ✅ Implement AlphaWriter (auto-extraction)
6. ✅ Implement FactorWriter
7. ✅ Implement dependency extraction from Expression
8. ✅ Implement LineageTracker

### Phase 3: Data Fetching (P2) - 1-2 weeks
1. ✅ Implement CCXTFetcher
2. ✅ Implement APIFetcher
3. ✅ Implement hive partitioning
4. ✅ Implement incremental update logic
5. ✅ Error handling & retry

### Phase 4: Advanced Features (P3-P4) - Future
1. [ ] PostgreSQL reader
2. [ ] ClickHouse reader
3. [ ] Fetch scheduler
4. [ ] Lineage visualization
5. [ ] Data quality checks

---

**Total Estimated Effort**: 5-8 weeks for P0-P2 (MVP)

-----
