# Alpha-Database 구현 가이드 (Implementation Guide)

## 1. 개요

이 문서는 alpha-database의 **구현 세부사항**을 다룹니다. 아키텍처 결정의 **"어떻게(How)"**에 초점을 맞춥니다.

**핵심 원칙**: alpha-database는 **데이터 CRUD와 쿼리**에만 집중합니다. 데이터 수집, 계산, Expression 직렬화는 다른 컴포넌트의 책임입니다.

## 2. P0: Config-Driven Data Loading 구현

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

### 2.8. Alpha-Canvas 통합

```python
# In alpha_canvas/core/facade.py (수정)

class AlphaCanvas:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        universe: Optional[xr.DataArray] = None,
        data_source: Optional['DataSource'] = None  # NEW!
    ):
        self.start_date = start_date
        self.end_date = end_date
        self._universe = universe
        
        # Dependency Injection
        if data_source is not None:
            self._data_source = data_source
        else:
            # Backward compatibility: use internal loader
            from .config import ConfigLoader
            from .data_loader import DataLoader
            self._config = ConfigLoader()
            self._data_loader = DataLoader()
            self._data_source = None
        
        # ... rest of initialization
```

```python
# In alpha_canvas/core/visitor.py (수정)

class EvaluateVisitor:
    def visit_field(self, node: Field) -> xr.DataArray:
        """Field 노드 방문 (데이터 로딩)."""
        # Check cache
        if node.name in self._canvas.db:
            result = self._canvas.db[node.name]
        else:
            # Use DataSource if available
            if self._canvas._data_source is not None:
                result = self._canvas._data_source.load_field(
                    node.name,
                    start_date=self._canvas.start_date,
                    end_date=self._canvas.end_date
                )
            else:
                # Backward compatibility: use internal loader
                result = self._canvas._data_loader.load_field(
                    node.name,
                    self._canvas.start_date,
                    self._canvas.end_date
                )
            
            # Cache
            self._canvas.db = self._canvas.db.assign({node.name: result})
        
        # Apply INPUT MASKING
        if self._universe_mask is not None:
            result = result.where(self._universe_mask, np.nan)
        
        return result
```

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

### 3.2. Alpha-Canvas Expression 직렬화 (alpha-canvas 책임)

**중요**: 이 코드는 **alpha-canvas**에 구현되어야 합니다. alpha-database는 직렬화된 결과(dict)만 저장합니다.

```python
# In alpha_canvas/core/expression.py (추가)

class Expression:
    """Base Expression class."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize Expression to dict (alpha-canvas 책임).
        
        Returns:
            Structured dict that can be saved as JSON
        """
        raise NotImplementedError("Subclass must implement to_dict()")
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Expression':
        """Deserialize dict to Expression (alpha-canvas 책임).
        
        Args:
            data: Serialized Expression dict
        
        Returns:
            Reconstructed Expression object
        """
        raise NotImplementedError("Must be implemented by subclasses")
    
    def get_field_dependencies(self) -> List[str]:
        """Extract Field dependencies (alpha-canvas 책임).
        
        Returns:
            List of field names this Expression depends on
        """
        deps = []
        
        if isinstance(self, Field):
            deps.append(self.name)
        elif hasattr(self, 'child'):
            deps.extend(self.child.get_field_dependencies())
        elif hasattr(self, 'left') and hasattr(self, 'right'):
            deps.extend(self.left.get_field_dependencies())
            deps.extend(self.right.get_field_dependencies())
        
        return list(set(deps))  # Deduplicate
```

```python
# In alpha_canvas/core/expression.py - Field 구현 예시

class Field(Expression):
    def __init__(self, name: str):
        self.name = name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'Field',
            'name': self.name
        }

# In alpha_canvas/ops/timeseries.py - TsMean 구현 예시

class TsMean(Expression):
    def __init__(self, child: Expression, window: int):
        self.child = child
        self.window = window
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'TsMean',
            'child': self.child.to_dict(),
            'window': self.window
        }

# Similar for all other Expression types...
```

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

**Implementation Version**: 2.0 (Simplified)  
**Last Updated**: 2025-01-23  
**Core Principle**: **alpha-database는 데이터 CRUD와 쿼리에만 집중합니다.**
