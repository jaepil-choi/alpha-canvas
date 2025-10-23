# Alpha-Database 구현 가이드 (Implementation Guide)

## 1. 개요

이 문서는 alpha-database의 **구현 세부사항**을 다룹니다. 아키텍처 결정의 **"어떻게(How)"**에 초점을 맞춥니다.

**핵심 원칙**: alpha-database는 **데이터 CRUD와 쿼리**에만 집중합니다. 데이터 수집, 계산, Expression 직렬화는 다른 컴포넌트의 책임입니다.

---

## 구현 현황 (Implementation Status)

### ✅ Phase 1: Config-Driven Data Loading (COMPLETE)
- ✅ ConfigLoader (독립적)
- ✅ DataLoader (pivoting)
- ✅ DataSource facade
- ✅ BaseReader interface
- ✅ ParquetReader (DuckDB backend)
- ✅ Plugin architecture (register_reader)
- ✅ Alpha-Canvas integration (dependency injection)
- ✅ 100% test coverage (40 tests passing)
- ✅ Experiment 20 validated (identical to old DataLoader)
- ✅ Showcase 18 (integration demonstration)

### ✅ Phase 1.5: Expression Serialization (COMPLETE - 2025-01-23)
- ✅ SerializationVisitor (Expression → JSON dict)
- ✅ DeserializationVisitor (dict → Expression)
- ✅ DependencyExtractor (field lineage)
- ✅ Convenience wrappers (to_dict, from_dict, get_field_dependencies)
- ✅ All 14 Expression types supported
- ✅ 33 comprehensive tests passing
- ✅ Round-trip validation complete
- ✅ Showcase 19 (serialization demonstration)

### 🔄 Phase 2: Data Storage (PLANNED)
- ⏳ DataWriter (field, alpha, factor)
- ⏳ MetaTable (catalog)
- ⏳ LineageTracker (explicit dependencies)

### 📝 Phase 3: Documentation & Migration (PLANNED)
- ⏳ End-to-end examples
- ⏳ Migration guide for users
- ⏳ Performance benchmarks

---

## 2. P0: Config-Driven Data Loading 구현 (✅ COMPLETE)

### 2.1. ConfigLoader 구현 (독립적)

**중요**: alpha-database는 alpha-canvas의 config에 의존하지 **않습니다**. 자체 ConfigLoader를 구현합니다.

**Location**: `alpha_database/core/config.py`

**책임**:
- `config/data.yaml` 파일 파싱
- Field 정의 (query, time_col, asset_col, value_col, reader type 등) 로드
- 설정 검증

**alpha-canvas config와의 차이**:
- alpha-database config는 **데이터 소스**만 정의
- alpha-canvas config는 **계산 로직**도 포함할 수 있음
- 두 config는 독립적으로 유지

### 2.2. DataLoader 구현

**Location**: `alpha_database/core/data_loader.py`

**책임**:
- Long 포맷 DataFrame을 Wide 포맷 DataArray로 피벗팅
- (date, security_id, value) → (time, asset) 변환
- xarray 좌표 설정

### 2.3. DataSource Facade 구현

```python
# alpha_database/core/data_source.py

from typing import Optional, Dict
import xarray as xr
from .config import ConfigLoader
from .data_loader import DataLoader
from ..readers import BaseReader, ParquetReader, CSVReader, ExcelReader

class DataSource:
    """통합 데이터 소스 facade with plugin support."""
    
    def __init__(self, config_path: str = 'config/data.yaml'):
        self._config = ConfigLoader(config_path)
        self._data_loader = DataLoader()
        self._readers: Dict[str, BaseReader] = self._init_core_readers()
    
    def _init_core_readers(self) -> Dict[str, BaseReader]:
        """Initialize core readers (built-in)."""
        return {
            'parquet': ParquetReader(),
            'csv': CSVReader(),
            'excel': ExcelReader(),
        }
    
    def register_reader(self, reader_type: str, reader: BaseReader):
        """Register custom reader (plugin).
        
        Args:
            reader_type: Name to use in config (e.g., 'fnguide_excel')
            reader: BaseReader instance
        """
        if reader_type in self._readers:
            raise ValueError(f"Reader '{reader_type}' already registered")
        self._readers[reader_type] = reader
    
    def load_field(
        self,
        field_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> xr.DataArray:
        """필드 데이터 로드 (Long → Wide 변환).
        
        Args:
            field_name: Field name from config
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            xarray.DataArray with (time, asset) dimensions
        """
        # 1. Config에서 필드 정의 로드
        field_config = self._config.get_field(field_name)
        
        # 2. Reader 선택 (plugin 지원)
        reader_type = field_config.get('reader', 'parquet')
        if reader_type not in self._readers:
            raise ValueError(
                f"Reader '{reader_type}' not found. "
                f"Available: {list(self._readers.keys())}"
            )
        reader = self._readers[reader_type]
        
        # 3. 파라미터 준비
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        # 4. Reader로 데이터 로드 (Long format)
        df_long = reader.read(
            query=field_config['query'],
            params=params
        )
        
        # 5. Long → Wide 피벗팅
        data_array = self._data_loader.pivot_to_xarray(
            df=df_long,
            time_col=field_config['time_col'],
            asset_col=field_config['asset_col'],
            value_col=field_config['value_col']
        )
        
        return data_array
```

### 2.4. BaseReader 인터페이스

```python
# alpha_database/readers/base.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseReader(ABC):
    """모든 Reader의 공통 인터페이스."""
    
    @abstractmethod
    def read(self, query: str, params: Dict[str, Any]) -> pd.DataFrame:
        """데이터 읽기 (Long 포맷 반환).
        
        Args:
            query: SQL template, file path, or other query format
            params: Runtime parameters (start_date, end_date, etc.)
        
        Returns:
            Long-format DataFrame with (time_col, asset_col, value_col)
        """
        pass
```

### 2.5. ParquetReader 구현

```python
# alpha_database/readers/parquet.py

import duckdb
import pandas as pd
from typing import Dict, Any
from .base import BaseReader

class ParquetReader(BaseReader):
    """DuckDB를 사용한 Parquet reader."""
    
    def read(self, query: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Parquet 파일 쿼리 실행.
        
        Args:
            query: SQL with {start_date}, {end_date} placeholders
            params: {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
        
        Returns:
            Long-format DataFrame
        """
        # 파라미터 치환
        formatted_query = query.format(**params)
        
        # DuckDB 쿼리 실행
        conn = duckdb.connect(':memory:')
        df = conn.execute(formatted_query).fetchdf()
        conn.close()
        
        return df
```

### 2.6. Plugin Architecture

**Core vs Plugin**:

**Core Readers** (alpha-database 패키지):
- ParquetReader, CSVReader, ExcelReader
- 모든 설치에 포함
- `alpha_database/readers/` 디렉토리

**Plugin Readers** (별도 패키지):
- `alpha-database-fnguide` (official plugin by alpha-database team)
- `alpha-database-bloomberg` (official plugin by alpha-database team)
- User-defined readers (community or custom)

**Official Plugin 예시: alpha-database-fnguide**

```python
# In alpha-database-fnguide package (별도 설치)
# Location: alpha_database_fnguide/fnguide_reader.py

from alpha_database.readers import BaseReader
import pandas as pd

class FnGuideExcelReader(BaseReader):
    """FnGuide 특수 포맷 Excel reader (official plugin)."""
    
    def read(self, query: str, params: Dict[str, Any]) -> pd.DataFrame:
        """FnGuide Excel 파일 읽기."""
        file_path = query
        
        # FnGuide-specific: header row skip
        header_row = params.get('header_row', 3)
        sheet_name = params.get('sheet_name', 0)
        
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header_row
        )
        
        # FnGuide-specific: column mapping
        date_col = params.get('date_col', '결산일')
        security_col = params.get('security_col', '종목코드')
        
        # Long format으로 변환
        # ... implementation ...
        
        return df_long
```

**Plugin 사용 (User code)**:

```python
# 1. Install plugin
# pip install alpha-database-fnguide

# 2. Import and register
from alpha_database import DataSource
from alpha_database_fnguide import FnGuideExcelReader

ds = DataSource(config_path='config/data.yaml')
ds.register_reader('fnguide_excel', FnGuideExcelReader())

# 3. Use in config
# config/data.yaml:
# fnguide_financials:
#   reader: fnguide_excel
#   file_path: 'data/fnguide.xlsx'
#   header_row: 3
```

**User-defined Reader 예시**:

```python
# User implements custom reader for proprietary format
class MyCustomReader(BaseReader):
    def read(self, query: str, params: dict) -> pd.DataFrame:
        # Custom logic
        ...

# Register
ds.register_reader('my_custom', MyCustomReader())
```

### 2.8. Alpha-Canvas 통합 (✅ IMPLEMENTED)

**Status**: ✅ Complete (Breaking Change - No Backward Compatibility)

**Implementation Date**: 2025-01-23

**Changes Made**:

```python
# In alpha_canvas/core/facade.py (IMPLEMENTED)

class AlphaCanvas:
    def __init__(
        self,
        data_source: 'DataSource',        # MANDATORY
        start_date: str,                  # MANDATORY
        end_date: Optional[str] = None,   # OPTIONAL
        config_dir: str = 'config',
        universe: Optional[Union[Expression, xr.DataArray]] = None
    ):
        """Initialize AlphaCanvas with DataSource injection.
        
        BREAKING CHANGE:
        - data_source: MANDATORY (no default)
        - start_date: MANDATORY (no default)
        - time_index, asset_index: REMOVED
        """
        self._data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        
        # Lazy panel initialization
        self._panel = None
        
        # Initialize evaluator with DataSource
        empty_ds = xr.Dataset()
        self._evaluator = EvaluateVisitor(empty_ds, data_source=data_source)
        self._evaluator._start_date = start_date
        self._evaluator._end_date = end_date
        
        # ... rest of initialization
```

```python
# In alpha_canvas/core/visitor.py (IMPLEMENTED)

class EvaluateVisitor:
    def __init__(self, data_source_ds: xr.Dataset, data_source=None):
        """Initialize with DataSource.
        
        Args:
            data_source_ds: xarray.Dataset for cached data
            data_source: Optional DataSource from alpha_database
        """
        self._data = data_source_ds
        self._data_source = data_source  # Changed from _data_loader
        self._start_date = None  # Set by AlphaCanvas
        self._end_date = None
    
    def visit_field(self, node: Field) -> xr.DataArray:
        """Field 노드 방문 (데이터 로딩)."""
        # Check cache
        if node.name in self._data:
            result = self._data[node.name]
        else:
            # Load via DataSource (MANDATORY)
            if self._data_source is None:
                raise RuntimeError(
                    f"Field '{node.name}' not found and no DataSource available"
                )
            
            result = self._data_source.load_field(
                node.name,
                start_date=self._start_date,
                end_date=self._end_date
            )
            
            # Cache
            self._data = self._data.assign({node.name: result})
        
        # Apply INPUT MASKING
        if self._universe_mask is not None:
            result = result.where(self._universe_mask, np.nan)
        
        self._cache_signal_weights_and_returns(f"Field_{node.name}", result)
        return result
```

**Migration Example**:

```python
# OLD (REMOVED):
rc = AlphaCanvas(
    time_index=pd.date_range('2024-01-01', periods=252),
    asset_index=['AAPL', 'GOOGL', 'MSFT']
)

# NEW (REQUIRED):
from alpha_database import DataSource

ds = DataSource('config')
rc = AlphaCanvas(
    data_source=ds,
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

**Validation Results**:
- ✅ 40/40 tests passing (100% success rate)
- ✅ Experiment 20: 100% identical results to old DataLoader
- ✅ Showcase 18: Full integration demonstrated
- ✅ TDD Red-Green cycle complete

---

## 3. P1: Data Storage 구현

### 3.1. DataWriter 구현 (Single Interface)

```python
# alpha_database/writers/data_writer.py

import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional, Union, Literal
from datetime import datetime

class DataWriter:
    """단일 데이터 저장 인터페이스 (field, alpha, factor 통합)."""
    
    def __init__(self, base_path: str = './data'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write(
        self,
        dataset_name: str,
        data: Union[xr.DataArray, Dict[str, xr.DataArray]],
        data_type: Literal['field', 'alpha', 'factor'],
        dependencies: List[str],  # Explicit! User-provided
        metadata: Optional[Dict] = None,
        field_name: Optional[str] = None  # Required for 'field' type
    ):
        """통합 저장 메서드.
        
        Args:
            dataset_name: Dataset name (e.g., 'fundamental', 'momentum_v1')
            data: DataArray or dict of DataArrays (for alpha)
            data_type: 'field', 'alpha', or 'factor'
            dependencies: Explicit list of field names (user-provided!)
            metadata: Optional metadata dict
            field_name: Field name (required for 'field' type)
        """
        if data_type == 'field':
            if field_name is None:
                raise ValueError("field_name required for data_type='field'")
            self._write_field(dataset_name, field_name, data, dependencies, metadata)
        
        elif data_type == 'alpha':
            self._write_alpha(dataset_name, data, dependencies, metadata)
        
        elif data_type == 'factor':
            self._write_factor(dataset_name, data, dependencies, metadata)
        
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def _write_field(
        self,
        dataset_name: str,
        field_name: str,
        data: xr.DataArray,
        dependencies: List[str],
        metadata: Optional[Dict]
    ):
        """필드 저장 (schema evolution 지원)."""
        dataset_path = self.base_path / f"{dataset_name}.parquet"
        
        # 1. Convert to long format
        df_new = self._to_long_format(data, field_name)
        
        # 2. Merge with existing data (if exists) - Schema Evolution
        if dataset_path.exists():
            df_existing = pd.read_parquet(dataset_path)
            
            # Upsert: drop old dates, append new data
            df_existing = df_existing[
                ~df_existing['date'].isin(df_new['date'])
            ]
            df_merged = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_merged = df_new
        
        # 3. Save
        df_merged.to_parquet(dataset_path, index=False)
        
        # 4. Update meta table
        from ..catalog import MetaTable
        meta = MetaTable()
        meta.upsert(
            name=f"{dataset_name}.{field_name}",
            type='field',
            location=str(dataset_path),
            dependencies=dependencies,  # User-provided!
            description=metadata.get('description', '') if metadata else '',
            tags=metadata.get('tags', []) if metadata else []
        )
    
    def _write_alpha(
        self,
        alpha_id: str,
        data: Dict[str, xr.DataArray],
        dependencies: List[str],
        metadata: Optional[Dict]
    ):
        """알파 저장 (auto-versioning).
        
        Args:
            alpha_id: Alpha identifier
            data: {'signal': DataArray, 'weights': DataArray, 'returns': DataArray}
            dependencies: User-provided list
            metadata: Must include 'expression' (pre-serialized by alpha-canvas)
        """
        # 1. Auto-version
        version = self._get_next_version(alpha_id)
        alpha_name = f"{alpha_id}_v{version}"
        alpha_dir = self.base_path / alpha_name
        alpha_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Save data files
        for key, array in data.items():
            if array is not None:
                array.to_netcdf(alpha_dir / f'{key}.nc')
        
        # 3. Save metadata (including pre-serialized Expression)
        meta_data = {
            'alpha_id': alpha_id,
            'version': version,
            'dependencies': dependencies,  # User-provided!
            'created': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        import json
        with open(alpha_dir / 'metadata.json', 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        # 4. Update meta table
        from ..catalog import MetaTable
        meta = MetaTable()
        meta.upsert(
            name=alpha_name,
            type='alpha',
            location=str(alpha_dir),
            dependencies=dependencies,  # User-provided!
            description=metadata.get('description', '') if metadata else '',
            tags=metadata.get('tags', []) if metadata else [],
            version=version
        )
    
    def _write_factor(
        self,
        factor_id: str,
        data: xr.DataArray,
        dependencies: List[str],
        metadata: Optional[Dict]
    ):
        """팩터 수익률 저장 (time series)."""
        factor_dir = self.base_path / factor_id
        factor_dir.mkdir(parents=True, exist_ok=True)
        
        # Save time series
        data.to_netcdf(factor_dir / 'returns.nc')
        
        # Save metadata
        meta_data = {
            'factor_id': factor_id,
            'dependencies': dependencies,  # User-provided!
            'created': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        import json
        with open(factor_dir / 'metadata.json', 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        # Update meta table
        from ..catalog import MetaTable
        meta = MetaTable()
        meta.upsert(
            name=factor_id,
            type='factor',
            location=str(factor_dir),
            dependencies=dependencies,  # User-provided!
            description=metadata.get('description', '') if metadata else '',
            tags=metadata.get('tags', []) if metadata else []
        )
    
    def _get_next_version(self, alpha_id: str) -> int:
        """다음 버전 번호 찾기."""
        existing = list(self.base_path.glob(f"{alpha_id}_v*"))
        if not existing:
            return 1
        
        versions = [
            int(p.name.split('_v')[1])
            for p in existing
        ]
        return max(versions) + 1
    
    def _to_long_format(self, data: xr.DataArray, field_name: str) -> pd.DataFrame:
        """Wide → Long 변환."""
        df = data.to_dataframe(name=field_name).reset_index()
        return df
```

### 3.2. Alpha-Canvas Expression 직렬화 (Visitor Pattern)

**중요**: 이 코드는 **alpha-canvas**에 구현되어야 합니다. alpha-database는 직렬화된 결과(dict)만 저장합니다.

**설계 원칙**: **Visitor Pattern**을 사용하여 Expression 직렬화를 수행합니다. 이는 Expression 클래스를 직렬화 로직으로부터 분리합니다.

```python
# In alpha_canvas/core/serialization.py (NEW FILE)

from typing import Dict, Any, List
from .expression import Expression, Field, Constant
from ..ops.timeseries import TsMean, TsAny
from ..ops.crosssection import Rank, CsQuantile
from ..ops.logical import And, Or

class SerializationVisitor:
    """Expression tree를 dict로 직렬화하는 visitor.
    
    이 visitor는 Expression tree를 순회하며 각 노드를
    JSON-serializable dict로 변환합니다.
    """
    
    def visit_field(self, node: Field) -> Dict[str, Any]:
        """Field 노드 직렬화."""
        return {
            'type': 'Field',
            'name': node.name
        }
    
    def visit_constant(self, node: Constant) -> Dict[str, Any]:
        """Constant 노드 직렬화."""
        return {
            'type': 'Constant',
            'value': node.value
        }
    
    def visit_ts_mean(self, node: TsMean) -> Dict[str, Any]:
        """TsMean 노드 직렬화."""
        return {
            'type': 'TsMean',
            'child': node.child.accept(self),  # Recursive
            'window': node.window
        }
    
    def visit_ts_any(self, node: TsAny) -> Dict[str, Any]:
        """TsAny 노드 직렬화."""
        return {
            'type': 'TsAny',
            'child': node.child.accept(self),
            'window': node.window
        }
    
    def visit_rank(self, node: Rank) -> Dict[str, Any]:
        """Rank 노드 직렬화."""
        return {
            'type': 'Rank',
            'child': node.child.accept(self)
        }
    
    def visit_cs_quantile(self, node: CsQuantile) -> Dict[str, Any]:
        """CsQuantile 노드 직렬화."""
        return {
            'type': 'CsQuantile',
            'child': node.child.accept(self),
            'q': node.q
        }
    
    # ... implement for all other Expression types
    # (And, Or, Add, Sub, Mul, Div, etc.)


class DeserializationVisitor:
    """Dict를 Expression tree로 역직렬화하는 visitor."""
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Expression:
        """Dict를 Expression으로 재구성.
        
        Args:
            data: 직렬화된 Expression dict
        
        Returns:
            재구성된 Expression 객체
        """
        expr_type = data['type']
        
        if expr_type == 'Field':
            return Field(data['name'])
        
        elif expr_type == 'Constant':
            return Constant(data['value'])
        
        elif expr_type == 'TsMean':
            child = DeserializationVisitor.from_dict(data['child'])
            return TsMean(child, window=data['window'])
        
        elif expr_type == 'TsAny':
            child = DeserializationVisitor.from_dict(data['child'])
            return TsAny(child, window=data['window'])
        
        elif expr_type == 'Rank':
            child = DeserializationVisitor.from_dict(data['child'])
            return Rank(child)
        
        elif expr_type == 'CsQuantile':
            child = DeserializationVisitor.from_dict(data['child'])
            return CsQuantile(child, q=data['q'])
        
        # ... handle all Expression types
        
        else:
            raise ValueError(f"Unknown expression type: {expr_type}")


class DependencyExtractor:
    """Expression tree에서 Field dependencies를 추출하는 visitor."""
    
    def __init__(self):
        self.dependencies: List[str] = []
    
    def visit_field(self, node: Field) -> None:
        """Field 노드 방문 시 의존성 추가."""
        self.dependencies.append(node.name)
    
    def visit_constant(self, node: Constant) -> None:
        """Constant는 의존성 없음."""
        pass
    
    def visit_ts_mean(self, node: TsMean) -> None:
        """TsMean의 child 순회."""
        node.child.accept(self)
    
    def visit_rank(self, node: Rank) -> None:
        """Rank의 child 순회."""
        node.child.accept(self)
    
    # ... implement for all operators
    
    @staticmethod
    def extract(expr: Expression) -> List[str]:
        """Expression에서 Field dependencies 추출.
        
        Args:
            expr: Expression tree
        
        Returns:
            List of unique field names
        """
        extractor = DependencyExtractor()
        expr.accept(extractor)
        return list(set(extractor.dependencies))  # Deduplicate
```

**Helper Functions for User Convenience**:

```python
# In alpha_canvas/core/expression.py (ADDED)

class Expression:
    """Base Expression class."""
    
    # ... existing methods ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize Expression to dict (convenience wrapper).
        
        Returns:
            JSON-serializable dict representation
        """
        from .serialization import SerializationVisitor
        return self.accept(SerializationVisitor())
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Expression':
        """Deserialize dict to Expression (convenience wrapper).
        
        Args:
            data: Serialized Expression dict
        
        Returns:
            Reconstructed Expression object
        """
        from .serialization import DeserializationVisitor
        return DeserializationVisitor.from_dict(data)
    
    def get_field_dependencies(self) -> List[str]:
        """Extract Field dependencies (convenience wrapper).
        
        Returns:
            List of field names this Expression depends on
        """
        from .serialization import DependencyExtractor
        return DependencyExtractor.extract(self)
```

**이점**:
1. ✅ **Separation of Concerns**: Expression 클래스는 순수 도메인 로직에만 집중
2. ✅ **확장성**: 새로운 직렬화 형식(YAML, Binary) 추가 시 새 visitor만 추가
3. ✅ **유지보수성**: 직렬화 로직이 한 곳에 집중됨
4. ✅ **테스트 용이성**: Visitor를 독립적으로 테스트 가능
5. ✅ **기존 인프라 활용**: EvaluateVisitor와 동일한 패턴 사용

### 3.3. 사용 패턴

```python
# User code (in Jupyter notebook)

from alpha_canvas import AlphaCanvas, Field, Expression
from alpha_canvas.ops import TsMean, Rank
from alpha_canvas.portfolio import DollarNeutralScaler
from alpha_database import DataWriter

# 1. Create alpha in alpha-canvas
rc = AlphaCanvas(start_date='2024-01-01', end_date='2024-12-31')
expr = Rank(TsMean(Field('returns'), window=5))
result = rc.evaluate(expr, scaler=DollarNeutralScaler())

# 2. alpha-canvas가 Expression 직렬화 및 의존성 추출
expr_dict = expr.to_dict()  # alpha-canvas method!
dependencies = expr.get_field_dependencies()  # alpha-canvas method!
# Returns: ['returns']

# 3. alpha-database에 저장 (pre-serialized data)
writer = DataWriter(base_path='./alphas')

writer.write(
    dataset_name='momentum_ma5_rank',
    data_type='alpha',
    data={
        'signal': rc.get_signal(step=2),
        'weights': rc.get_weights(step=2),
        'returns': rc.get_port_return(step=2)
    },
    dependencies=dependencies,  # From alpha-canvas!
    metadata={
        'expression': expr_dict,  # Pre-serialized by alpha-canvas!
        'expression_str': str(expr),
        'description': 'Momentum strategy with 5-day MA and rank',
        'tags': ['momentum', 'mean-reversion']
    }
)
```

### 3.4. Meta Table 구현

```python
# alpha_database/catalog/meta_table.py

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

class MetaTable:
    """중앙 메타데이터 레지스트리."""
    
    def __init__(self, path: str = './data/meta_table.parquet'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.path.exists():
            self._init_table()
    
    def _init_table(self):
        """빈 테이블 초기화."""
        df = pd.DataFrame({
            'name': pd.Series(dtype='str'),
            'type': pd.Series(dtype='str'),
            'location': pd.Series(dtype='str'),
            'dependencies': pd.Series(dtype='object'),  # List[str]
            'created': pd.Series(dtype='datetime64[ns]'),
            'updated': pd.Series(dtype='datetime64[ns]'),
            'description': pd.Series(dtype='str'),
            'tags': pd.Series(dtype='object'),  # List[str]
            'version': pd.Series(dtype='Int64')
        })
        df.to_parquet(self.path, index=False)
    
    def upsert(
        self,
        name: str,
        type: str,
        location: str,
        dependencies: List[str],  # User-provided!
        description: str = '',
        tags: List[str] = None,
        version: Optional[int] = None
    ):
        """레코드 추가 또는 업데이트."""
        df = pd.read_parquet(self.path)
        
        now = datetime.now()
        
        # Check if exists
        existing = df[df['name'] == name]
        
        if len(existing) > 0:
            # Update
            df.loc[df['name'] == name, 'updated'] = now
            df.loc[df['name'] == name, 'location'] = location
            df.loc[df['name'] == name, 'dependencies'] = [dependencies]
            df.loc[df['name'] == name, 'description'] = description
            if tags:
                df.loc[df['name'] == name, 'tags'] = [tags]
        else:
            # Insert
            new_row = {
                'name': name,
                'type': type,
                'location': location,
                'dependencies': [dependencies],
                'created': now,
                'updated': now,
                'description': description,
                'tags': [tags] if tags else [[]],
                'version': version
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        df.to_parquet(self.path, index=False)
    
    def query(self, type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """테이블 쿼리."""
        df = pd.read_parquet(self.path)
        
        if type:
            df = df[df['type'] == type]
        
        return df
    
    def find_dependents(self, field_name: str) -> List[str]:
        """특정 필드에 의존하는 모든 엔티티 찾기."""
        df = pd.read_parquet(self.path)
        
        dependents = []
        for idx, row in df.iterrows():
            if field_name in row['dependencies']:
                dependents.append(row['name'])
        
        return dependents
    
    def search(self, tags__contains: Optional[str] = None) -> pd.DataFrame:
        """Tag 검색."""
        df = pd.read_parquet(self.path)
        
        if tags__contains:
            mask = df['tags'].apply(
                lambda tags: tags__contains in tags if tags else False
            )
            df = df[mask]
        
        return df
```

---

## 4. P2: Lineage Tracking 구현 (Explicit Dependencies Only)

### 4.1. LineageTracker 구현

```python
# alpha_database/catalog/lineage.py

import pandas as pd
from pathlib import Path
from typing import List, Set, Dict
from .meta_table import MetaTable

class LineageTracker:
    """의존성 그래프 추적 (user-provided dependencies only)."""
    
    def __init__(self, meta_table_path: str = './data/meta_table.parquet'):
        self.meta = MetaTable(meta_table_path)
        self.df = pd.read_parquet(meta_table_path)
    
    def find_dependents(self, field_name: str) -> List[str]:
        """특정 필드에 의존하는 모든 엔티티 (1-level).
        
        Note:
            User-provided dependencies만 사용 (Expression 파싱 없음)
        """
        return self.meta.find_dependents(field_name)
    
    def find_dependencies(self, entity_name: str) -> List[str]:
        """특정 엔티티가 의존하는 모든 필드 (1-level, explicit only).
        
        Args:
            entity_name: Entity to analyze
        
        Returns:
            List of field names (user-provided dependencies only)
        """
        row = self.df[self.df['name'] == entity_name]
        if len(row) == 0:
            return []
        return row.iloc[0]['dependencies']
    
    def get_impact(self, field_name: str) -> List[str]:
        """Impact analysis: 데이터 변경 시 영향받는 엔티티.
        
        Returns:
            List of entity names that depend on field_name
        """
        # Simple 1-level impact (no recursive traversal)
        return self.find_dependents(field_name)
    
    def get_lineage(self, entity_name: str) -> Dict[str, List[str]]:
        """전체 계보 경로 반환 (explicit links only).
        
        Returns:
            Dict mapping entity to its user-provided dependencies
        """
        lineage = {}
        lineage[entity_name] = self.find_dependencies(entity_name)
        return lineage
```

**중요**: 이 구현은 매우 단순합니다. Expression 파싱이나 재귀적 의존성 계산을 하지 않습니다. 사용자가 `write()` 시 제공한 `dependencies`만 사용합니다.

---

## 5. 테스트 전략

### 5.1. Unit Tests

**DataSource**:
- Config 파싱 테스트
- Reader 선택 로직 테스트
- Plugin 등록 테스트
- 파라미터 치환 테스트

**Readers**:
- Long 포맷 반환 검증
- 파라미터 처리 테스트
- 에러 처리 테스트

**DataWriter**:
- 데이터 저장 검증 (field, alpha, factor)
- Schema evolution 테스트
- Meta table 업데이트 검증
- Versioning 테스트

**MetaTable**:
- CRUD 작업 테스트
- 쿼리 테스트
- Dependency 검색 테스트

### 5.2. Integration Tests

**Alpha-Canvas 통합**:
- DataSource injection 테스트
- Field 로딩 검증
- Backward compatibility 검증

**End-to-End**:
```python
def test_e2e_alpha_save_and_load():
    # 1. Create alpha in alpha-canvas
    rc = AlphaCanvas(data_source=ds, ...)
    expr = Rank(TsMean(Field('returns'), window=5))
    result = rc.evaluate(expr, scaler=scaler)
    
    # 2. alpha-canvas가 직렬화
    expr_dict = expr.to_dict()
    deps = expr.get_field_dependencies()
    
    # 3. alpha-database에 저장
    writer.write(..., dependencies=deps, metadata={'expression': expr_dict})
    
    # 4. 재로드 및 재현
    reader = DataReader(...)
    alpha_data = reader.read('momentum_v1')
    expr_reconstructed = Expression.from_dict(alpha_data['metadata']['expression'])
    
    # 5. 재실행
    result2 = rc.evaluate(expr_reconstructed, scaler=scaler)
    
    # 6. 검증
    assert result.equals(result2)
```

### 5.3. Performance Tests

**DataLoader**:
- 대용량 Parquet 쿼리 성능
- 피벗팅 성능
- DuckDB 최적화 검증

**DataWriter**:
- 대용량 저장 성능
- Schema evolution 성능

---

## 6. 마이그레이션 가이드

### 6.1. Phase 1: DataSource 구현

1. `alpha_database/core/` 구현
2. Reader 패밀리 구현
3. Plugin 시스템 구현
4. 테스트 작성 및 검증

### 6.2. Phase 2: Alpha-Canvas 통합

1. `AlphaCanvas.__init__()` 수정 (data_source 파라미터)
2. `EvaluateVisitor.visit_field()` 수정
3. **Expression 직렬화 메서드 추가** (`to_dict()`, `from_dict()`, `get_field_dependencies()`)
4. Backward compatibility 유지
5. 기존 테스트 모두 통과 확인

### 6.3. Phase 3: Storage 구현

1. DataWriter 구현 (단일 인터페이스)
2. Meta Table 구현
3. LineageTracker 구현 (explicit only)
4. 테스트 작성

### 6.4. Phase 4: 검증 및 마이그레이션

1. End-to-end 테스트
2. 성능 벤치마크
3. alpha-canvas 내장 loader 제거 (Phase 3)

---

## 7. 아키텍처 단순화 요약

### 제외된 구현

**1. Data Fetching (CCXTFetcher, APIFetcher)**
- **이유**: ETL 파이프라인의 역할
- **대안**: 사용자가 직접 데이터 수집 스크립트 작성

**2. Expression Auto-Serialization**
- **이유**: alpha-canvas의 책임
- **대안**: alpha-canvas가 `to_dict()` 메서드 제공, alpha-database는 결과만 저장

**3. Automatic Dependency Extraction**
- **이유**: Expression 파싱은 복잡하고 강결합 유발
- **대안**: 사용자가 `write()` 시 명시적으로 `dependencies` 제공, alpha-canvas가 `get_field_dependencies()` helper 제공

**4. Triple-Cache Management**
- **이유**: 계산 캐시는 alpha-canvas의 책임
- **대안**: alpha-canvas가 캐시 관리, alpha-database는 최종 결과만 저장

**5. Specialized Readers (Built-in FnGuide, Bloomberg 등)**
- **이유**: 유지보수 부담, 확장성 제한
- **대안**: Plugin architecture

### 구현된 핵심 기능

1. ✅ **DataSource Facade** (with plugin support)
2. ✅ **Core Readers** (Parquet, CSV, Excel)
3. ✅ **Single DataWriter** (field, alpha, factor 통합)
4. ✅ **Meta Table** (simple Parquet table)
5. ✅ **LineageTracker** (explicit dependencies only)

---

## 8. 다음 단계 (Next Steps)

### 8.1. Phase 1 완료 검증 ✅

**완료된 작업**:
- ✅ ConfigLoader, DataLoader, DataSource 구현
- ✅ BaseReader, ParquetReader 구현
- ✅ Plugin architecture 구현
- ✅ AlphaCanvas 통합 (breaking change)
- ✅ 40 tests passing (100% success rate)
- ✅ Experiment 20 validated
- ✅ Showcase 17 & 18 completed
- ✅ Committed: `feat: integrate DataSource into AlphaCanvas`

### 8.2. Phase 2 시작 전 정리 작업

**권장 작업 순서**:

1. **Old DataLoader 제거** (alpha-canvas 내부):
   - `src/alpha_canvas/core/data_loader.py` 삭제
   - 더 이상 사용되지 않음 (DataSource로 완전 대체)
   - 모든 테스트 여전히 통과하는지 확인

**Note**: Remaining showcases (1-16) 업데이트는 선택사항입니다. Showcases는 temporal completeness를 위한 것이며, 필수 작업이 아닙니다. README.md 업데이트도 필요하지 않습니다.

### 8.3. Phase 2: Data Storage 구현 (다음 큰 작업)

**Phase 2 목표**:
- DataWriter 구현 (field, alpha, factor 저장)
- MetaTable 구현 (카탈로그)
- LineageTracker 구현 (의존성 추적)

**Phase 2 전제조건**:
- ✅ **Expression 직렬화** (COMPLETED - 2025-01-23):
  - ✅ `SerializationVisitor` (Expression → dict)
  - ✅ `DeserializationVisitor` (dict → Expression)
  - ✅ `DependencyExtractor` (Extract Field dependencies)
  - ✅ 모든 14개 Expression 타입 지원 (Field, Constant, TsMean, TsAny, Rank, CsQuantile, comparison operators, logical operators)
  - ✅ Convenience wrappers: `Expression.to_dict()`, `Expression.from_dict()`, `Expression.get_field_dependencies()`
  - ✅ 33 tests passing, round-trip validation complete
  - ✅ Showcase 19 demonstrating all capabilities

**Phase 2 시작 전 확인사항**:
- [x] Phase 1 완전히 검증됨
- [ ] Old DataLoader 제거됨
- [ ] Documentation 업데이트됨
- [x] Expression 직렬화 구현 완료됨 (2025-01-23)

### 8.4. 즉시 실행 가능한 작업 (Quick Wins)

**Option 1: Cleanup & Documentation** (추천)
1. Old DataLoader 제거
2. Showcase 업데이트 (1-16)
3. README 업데이트
4. Commit: `chore: remove old DataLoader and update showcases`

**Option 2: Expression Serialization** ✅ COMPLETED (2025-01-23)
1. ✅ `SerializationVisitor` 구현 (Expression → dict)
2. ✅ `DeserializationVisitor` 구현 (dict → Expression)
3. ✅ `DependencyExtractor` 구현 (Field dependencies)
4. ✅ 모든 14개 Expression 타입에 visitor 메서드 추가
5. ✅ Convenience wrappers 구현 (`to_dict()`, `from_dict()`, `get_field_dependencies()`)
6. ✅ Unit tests 작성 (33 tests passing)
7. ✅ Commit: `feat: implement visitor-based Expression serialization`

**Option 3: Phase 2 시작** (바로 진행)
1. DataWriter 구현 시작
2. Field 저장 기능부터 구현
3. MetaTable skeleton 구현
4. TDD로 진행

---

**Implementation Version**: 2.0 (Simplified)  
**Last Updated**: 2025-01-23 (Phase 1 Complete)  
**Core Principle**: **alpha-database는 데이터 CRUD와 쿼리에만 집중합니다.**
