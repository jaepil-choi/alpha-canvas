# 3. Implementation Guide

이 문서는 alpha-canvas의 구체적인 구현 방법론, 인터페이스 설계, 그리고 개발 표준을 정의합니다.

## 3.1. 프로젝트 구조

```text
alpha-canvas/
├── config/                      # 타입별 설정 파일
│   ├── data.yaml               # 데이터 필드 정의
│   ├── db.yaml                 # 데이터베이스 연결 설정 (선택적)
│   └── compute.yaml            # 계산 관련 설정 (선택적)
├── src/
│   └── alpha_canvas/
│       ├── __init__.py
│       ├── core/
│       │   ├── facade.py       # AlphaCanvas (rc) 퍼사드 클래스
│       │   ├── expression.py   # Expression 컴포짓 트리
│       │   ├── visitor.py      # EvaluateVisitor 패턴 (타입 검사 포함)
│       │   └── config.py       # ConfigLoader
│       ├── ops/                # 연산자 (ts_mean, rank, etc.)
│       │   ├── __init__.py
│       │   ├── timeseries.py   # ts_mean, ts_sum, etc. (다형성 연산자)
│       │   ├── crosssection.py # cs_rank 등 (Panel 전용 연산자)
│       │   ├── classification.py # cs_quantile, cs_cut (분류기/축 생성)
│       │   ├── transform.py    # group_neutralize, etc.
│       │   └── tensor.py       # 미래 확장용 (MVP에서는 비어있음)
│       ├── portfolio/          # 포트폴리오 구성
│       │   ├── __init__.py
│       │   ├── base.py         # WeightScaler 추상 베이스 클래스
│       │   └── strategies.py   # GrossNetScaler, DollarNeutralScaler, LongOnlyScaler
│       ├── analysis/
│       │   ├── pnl.py          # PnLTracer
│       │   └── metrics.py      # 성과 지표 계산
│       └── utils/
│           ├── accessor.py     # Property 접근자 (data, axis, rules)
│           └── mask.py         # 마스크 헬퍼
├── experiments/                # 실험 스크립트
├── tests/                      # 테스트
└── docs/
    └── vibe_coding/
        ├── prd.md
        ├── architecture.md
        └── implementation.md   # 이 문서
```

## 3.2. 핵심 인터페이스 설계

### 3.2.1. 초기화 및 설정

```python
from alpha_canvas import AlphaCanvas

# config/ 디렉토리의 모든 YAML 파일을 자동 로드
rc = AlphaCanvas()

# 또는 특정 config 디렉토리 지정
rc = AlphaCanvas(config_dir='./custom_config')
```

**구현 요구사항:**

- `AlphaCanvas.__init__()` 내부에서 `ConfigLoader`를 생성하고 `config/` 디렉토리의 모든 `.yaml` 파일을 로드합니다.
- `ConfigLoader`는 `data.yaml`, `db.yaml` 등을 각각 파싱하여 내부 dict에 저장합니다.

### 3.2.2. 코어 데이터 모델 구현 (Core Data Model Implementation)

#### A. `AlphaCanvas.add_data()` 구현 (`facade.py`)

```python
def add_data(self, name: str, data: Union[Expression, xr.DataArray]) -> None:
    """데이터 변수를 Dataset에 추가 (Expression 또는 DataArray 지원)"""
    
    # Case 1: Expression 평가 (일반적인 경로)
    if isinstance(data, Expression):
        self.rules[name] = data  # Expression 저장 (재평가 가능하도록)
        result_array = self._evaluator.evaluate(data)  # Visitor로 평가
        self.db = self.db.assign({name: result_array})  # data_vars에 추가
    
    # Case 2: DataArray 직접 주입 (Open Toolkit: Inject)
    elif isinstance(data, xr.DataArray):
        # 외부에서 생성한 데이터 주입 (Visitor 건너뛰기)
        self.db = self.db.assign({name: data})
    
    else:
        raise TypeError(f"data must be Expression or DataArray, got {type(data)}")
```

**핵심 사항:**

- `xarray.Dataset.assign()`을 사용하여 Data Variable로 추가
- `Expression`과 `DataArray` 모두 지원 (오버로딩)
- Open Toolkit 철학: 외부 계산 결과를 seamlessly inject

#### B. `rc.db` 프로퍼티 (Open Toolkit: Eject)

```python
@property
def db(self) -> xr.Dataset:
    """순수 xarray.Dataset 반환 (Jupyter eject용)"""
    return self._dataset  # 내부 Dataset을 그대로 노출
```

**핵심 사항:**

- 래핑 없이 순수 `xarray.Dataset` 반환
- 사용자는 `pure_ds = rc.db`로 꺼내서 scipy/statsmodels 사용 가능

#### C. 유니버스 마스킹 (Universe Masking) ✅ **IMPLEMENTED**

**요구사항**: 초기화 시 유니버스를 설정하고, 모든 데이터와 연산에 자동 적용

```python
# AlphaCanvas 초기화 with universe
rc = AlphaCanvas(
    start_date='2024-01-01',
    end_date='2024-12-31',
    universe=price > 5.0  # Boolean DataArray
)

# 또는 Expression으로 설정
rc = AlphaCanvas(
    start_date='2024-01-01',
    end_date='2024-12-31',
    universe=Field('univ500')  # Field Expression (미래 확장)
)

# 유니버스 확인 (read-only)
print(f"Universe coverage: {rc.universe.sum().values} positions")
```

**구현 세부사항**:

**1. AlphaCanvas에 universe 파라미터 추가**:
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
        # ... 기존 초기화 ...
        
        # Universe mask 초기화 (불변)
        self._universe_mask: Optional[xr.DataArray] = None
        if universe is not None:
            self._set_initial_universe(universe)
    
    def _set_initial_universe(self, universe: Union[Expression, xr.DataArray]) -> None:
        """유니버스 마스크를 초기화 시 한 번만 설정 (불변)."""
        # Expression 평가 (e.g., Field('univ500'))
        if isinstance(universe, Expression):
            universe_data = self._evaluator.evaluate(universe)
        else:
            universe_data = universe
        
        # Shape 검증
        expected_shape = (
            len(self._panel.db.coords['time']), 
            len(self._panel.db.coords['asset'])
        )
        if universe_data.shape != expected_shape:
            raise ValueError(
                f"Universe mask shape {universe_data.shape} doesn't match "
                f"data shape {expected_shape}"
            )
        
        # Dtype 검증
        if universe_data.dtype != bool:
            raise TypeError(f"Universe must be boolean, got {universe_data.dtype}")
        
        # 불변 저장
        self._universe_mask = universe_data
        
        # Evaluator에 전파 (자동 적용 위해)
        self._evaluator._universe_mask = self._universe_mask
    
    @property
    def universe(self) -> Optional[xr.DataArray]:
        """유니버스 마스크 조회 (read-only)."""
        return self._universe_mask
```

**2. EvaluateVisitor에 이중 마스킹 구현**:
```python
class EvaluateVisitor:
    def __init__(self, data_source: xr.Dataset, data_loader=None):
        self._data = data_source
        self._data_loader = data_loader
        self._universe_mask: Optional[xr.DataArray] = None  # AlphaCanvas가 설정
        self._cache: Dict[int, Tuple[str, xr.DataArray]] = {}
        self._step_counter = 0
    
    def visit_field(self, node) -> xr.DataArray:
        """Field 노드 방문 with INPUT MASKING."""
        # 필드 로드 (캐시 또는 DataLoader)
        if node.name in self._data:
            result = self._data[node.name]
        else:
            if self._data_loader is None:
                raise RuntimeError(f"Field '{node.name}' not found")
            result = self._data_loader.load_field(node.name)
            self._data = self._data.assign({node.name: result})
        
        # INPUT MASKING: 필드 검색 시 유니버스 적용
        if self._universe_mask is not None:
            result = result.where(self._universe_mask, np.nan)
        
        self._cache_result(f"Field_{node.name}", result)
        return result
    
    def visit_operator(self, node) -> xr.DataArray:
        """연산자 방문 with OUTPUT MASKING."""
        # 1. 순회: 자식 평가 (이미 마스킹됨)
        child_result = node.child.accept(self)
        
        # 2. 위임: 연산자의 compute() 호출
        result = node.compute(child_result)
        
        # 3. OUTPUT MASKING: 연산 결과에 유니버스 적용
        if self._universe_mask is not None:
            result = result.where(self._universe_mask, np.nan)
        
        # 4. 캐싱
        operator_name = node.__class__.__name__
        self._cache_result(operator_name, result)
        
        return result
```

**3. add_data()에서 주입 데이터 마스킹**:
```python
def add_data(self, name: str, data: Union[Expression, xr.DataArray]) -> None:
    """데이터 추가 with 유니버스 마스킹."""
    if isinstance(data, Expression):
        # Expression 경로 - Evaluator가 자동 마스킹
        self.rules[name] = data
        result = self._evaluator.evaluate(data)
        self._panel.add_data(name, result)
        
        # Evaluator 재동기화 시 유니버스 보존
        self._evaluator = EvaluateVisitor(self._panel.db, self._data_loader)
        if self._universe_mask is not None:
            self._evaluator._universe_mask = self._universe_mask
    
    elif isinstance(data, xr.DataArray):
        # DataArray 직접 주입 - 여기서 마스킹
        if self._universe_mask is not None:
            data = data.where(self._universe_mask, float('nan'))
        
        self._panel.add_data(name, data)
        self._evaluator = EvaluateVisitor(self._panel.db, self._data_loader)
        if self._universe_mask is not None:
            self._evaluator._universe_mask = self._universe_mask
```

**핵심 사항**:
- **불변성**: 유니버스는 초기화 시 한 번만 설정, 변경 불가
- **이중 마스킹**: Field 입력 + Operator 출력 모두 마스킹 (신뢰 체인)
- **Open Toolkit**: 주입된 DataArray도 자동 마스킹
- **성능**: 13.6% 오버헤드 (xarray lazy evaluation으로 무시 가능)

---

#### D. `rc.data` Accessor 구현 (Selector Interface) ✅ **IMPLEMENTED**

**설계 철학**: Expression 기반 필드 접근으로 지연 평가 및 유니버스 안전성 보장

```python
from alpha_canvas.core.expression import Field


class DataAccessor:
    """rc.data accessor that returns Field Expressions.
    
    This enables Expression-based data access:
        rc.data['field_name'] → Field('field_name')
        rc.data['size'] == 'small' → Equals(Field('size'), 'small')
    
    Field Expressions remain lazy until explicitly evaluated,
    ensuring universe masking through the Visitor pattern.
    """
    
    def __getitem__(self, field_name: str) -> Field:
        """Return Field Expression for the given field name.
        
        Args:
            field_name: Name of the field to access
            
        Returns:
            Field Expression wrapping the field name
            
        Raises:
            TypeError: If field_name is not a string
        """
        if not isinstance(field_name, str):
            raise TypeError(
                f"Field name must be string, got {type(field_name).__name__}"
            )
        
        return Field(field_name)
    
    def __getattr__(self, name: str):
        """Prevent attribute access - only item access allowed.
        
        This ensures a single, consistent access pattern.
        """
        raise AttributeError(
            f"DataAccessor does not support attribute access. "
            f"Use rc.data['{name}'] instead of rc.data.{name}"
        )
```

**AlphaCanvas 통합**:

```python
class AlphaCanvas:
    def __init__(self, ...):
        # ... existing init ...
        self._data_accessor = DataAccessor()  # Create once, reuse
    
    @property
    def data(self) -> DataAccessor:
        """Access data fields as Field Expressions."""
        return self._data_accessor
```

**핵심 사항:**

- ✅ **Expression 반환**: `rc.data['size']` → `Field('size')` (lazy)
- ✅ **Lazy 평가**: 명시적 `rc.evaluate()` 호출 전까지 평가 안 됨
- ✅ **유니버스 안전**: 모든 Expression은 Visitor를 통해 평가되어 유니버스 마스킹 보장
- ✅ **Composable**: `ts_mean(rc.data['returns'], 10)` 같은 체이닝 가능
- ✅ **Item access only**: `rc.data['field']`만 지원, `rc.data.field`는 에러
- ✅ **통합**: Phase 7A Boolean Expression과 완벽 통합

**사용 패턴**:

```python
# ✅ Correct pattern (Expression-based)
mask = rc.data['size'] == 'small'  # Returns Equals Expression
result = rc.evaluate(mask)         # Evaluates with universe masking

# ✅ Complex pattern
mask = (rc.data['size'] == 'small') & (rc.data['momentum'] == 'high')
result = rc.evaluate(mask)

# ❌ Wrong pattern (not supported)
mask = rc.data.size == 'small'  # AttributeError
```

### 3.2.3. Interface A: Formula-based (Excel-like)

```python
from alpha_canvas.ops import ts_mean, rank, group_neutralize, Field

# 1. 간단한 헬퍼 함수 스타일 (즉시 평가)
returns_10d = rc.ts_mean('returns', 10)  
# rc.db['returns_10d']에 DataArray 저장

# 2. 복잡한 Expression 정의 (지연 평가)
alpha_expr = group_neutralize(
    rank(ts_mean(Field('returns'), 10)),
    group_by='subindustry'
)

# 3. Expression을 변수로 등록
rc.add_data_var('alpha1', alpha_expr)

# 4. 데이터 접근 (evaluated data)
alpha1_data = rc.db['alpha1']  # xarray.DataArray (T, N)
```

**구현 요구사항:**

- `Field('returns')`: `ConfigLoader`에서 `config/data.yaml`의 `returns` 정의를 참조하는 Leaf Expression
- `ts_mean()`, `rank()` 등: Composite Expression 노드를 생성
- `rc.add_data_var()`: Expression을 `rc.rules`에 등록하고, `EvaluateVisitor`로 평가하여 `rc.db`에 저장

### 3.2.3. Interface B: Selector-based (NumPy-like)

```python
# 1. 시그널 캔버스 초기화
rc.init_signal_canvas('my_alpha')  
# rc.db['my_alpha']에 (T, N) 영행렬 생성

# 2. 데이터 등록
rc.add_data('mcap', Field('market_cap'))
rc.add_data('ret', Field('returns'))
rc.add_data('vol', Field('volume'))

# 3. 분류 데이터 정의 - 레이블 기반 버킷
rc.add_data('size', cs_quantile(rc.data['mcap'], bins=3, labels=['small', 'mid', 'big']))
rc.add_data('momentum', cs_quantile(rc.data['ret'], bins=2, labels=['low', 'high']))
rc.add_data('surge', ts_any(rc.data['ret'] > 0.3, window=252))  # Boolean

# 4. 비교 연산으로 Boolean 마스크 생성
mask_long = (rc.data['size'] == 'small') & (rc.data['momentum'] == 'high') & (rc.data['surge'] == True)
mask_short = (rc.data['size'] == 'big') & (rc.data['momentum'] == 'low')

# 5. NumPy-style 할당
rc['my_alpha'][mask_long] = 1.0
rc['my_alpha'][mask_short] = -1.0

# 또는 현재 활성 캔버스에 직접 할당
rc[mask_long] = 1.0

# 6. 최종 시그널 접근 (evaluated data)
my_alpha = rc.db['my_alpha']  # xarray.DataArray (T, N)
```

**구현 요구사항:**

- `rc.add_data('size', expr)`: Expression을 평가하고 `rc.db.assign({'size': result})`로 data_vars에 추가
- `rc.data['size'] == 'small'`:
  1. `rc.data['size']` → `Field('size')` Expression 반환
  2. `Field('size') == 'small'` → `Equals(Field('size'), 'small')` Expression 반환
  3. Expression은 lazy하게 유지, `rc.evaluate(expr)`로 평가
- `rc[mask] = value`: `xr.where(mask, value, rc.db[current_canvas])`로 할당 (미구현)

### 3.2.4. Interface C: Selective Traceability (Integer-Based Steps)

```python
# 복잡한 Expression 정의
complex_alpha = group_neutralize(
    rank(ts_mean(Field('returns'), 5)),
    group_by='subindustry'
)

rc.add_data_var('complex_alpha', complex_alpha)

# Expression 트리 구조 (depth-first 순서):
# step 0: Field('returns')
# step 1: ts_mean(Field('returns'), 5)
# step 2: rank(...)
# step 3: group_neutralize(...)

# 1. 특정 단계만 추적
pnl_step1 = rc.trace_pnl('complex_alpha', step=1)
# {'sharpe': 0.7, 'total_pnl': 150, 'cumulative_returns': [...]}

# 2. 모든 단계 추적
pnl_all = rc.trace_pnl('complex_alpha')  # step=None (default)
# {
#   0: {'step_name': 'Field_returns', 'sharpe': 0.5, ...},
#   1: {'step_name': 'ts_mean', 'sharpe': 0.7, ...},
#   2: {'step_name': 'rank', 'sharpe': 0.6, ...},
#   3: {'step_name': 'group_neutralize', 'sharpe': 0.8, ...}
# }

# 3. 중간 데이터 직접 접근
intermediate = rc.get_intermediate('complex_alpha', step=1)
# xarray.DataArray (T, N) - ts_mean 적용 후 데이터

# 4. 복잡한 Expression 예시 (병렬 연산)
combo_alpha = ts_mean(Field('returns'), 3) + rank(Field('market_cap'))
rc.add_data_var('combo', combo_alpha)

# step 0: Field('returns')
# step 1: ts_mean(Field('returns'), 3)
# step 2: Field('market_cap')
# step 3: rank(Field('market_cap'))
# step 4: add(step1, step3)

pnl_step4 = rc.trace_pnl('combo', step=4)  # 최종 결과
```

**구현 요구사항:**

- `EvaluateVisitor.cache` 구조: `dict[str, dict[int, tuple[str, xr.DataArray]]]`
  - 외부 키: 변수명 (e.g., `'complex_alpha'`)
  - 내부 키: 정수 step 인덱스 (0부터 시작)
  - 값: `(노드_이름, DataArray)` 튜플 (디버깅용 메타데이터 포함)
- `EvaluateVisitor._step_counter`: 현재 step 인덱스를 추적하는 내부 카운터
- `EvaluateVisitor`는 Expression 트리를 **깊이 우선 탐색(depth-first)** 으로 순회하며 각 노드의 반환값을 캐시에 저장
- `rc.trace_pnl(var, step=None)`:
  - `step=None`: 모든 단계의 캐시 데이터를 `PnLTracer`에 전달
  - `step=1`: 해당 단계만 전달
- `rc.get_intermediate(var, step)`: `rc._evaluator.cache[var][step][1]` 반환 (DataArray 부분)

### 3.2.5. 핵심 활용 패턴: 팩터 수익률 계산

#### A. 독립 이중 정렬 (Independent Double Sort) - Fama-French SMB

```python
from alpha_canvas.ops import cs_quantile, Field

# 1. 데이터 등록
rc.add_data('mcap', Field('market_cap'))
rc.add_data('btm', Field('book_to_market'))

# 2. 독립 정렬: 전체 유니버스에서 각각 quantile 계산
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small', 'big']))
rc.add_axis('value', cs_quantile(rc.data.btm, bins=3, labels=['low', 'mid', 'high']))

# 3. SMB 포트폴리오 구성
rc.init_signal_canvas('smb')
rc[rc.axis.size['small']] = 1.0   # Long all small stocks
rc[rc.axis.size['big']] = -1.0    # Short all big stocks

# 4. 팩터 수익률 추적
smb_returns = rc.trace_pnl('smb')
print(f"SMB Sharpe: {smb_returns['sharpe']:.2f}")
```

#### B. 종속 이중 정렬 (Dependent Double Sort) - Fama-French HML

```python
# 1. 첫 번째 정렬: Size
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small', 'big']))

# 2. 종속 정렬: 각 Size 그룹 내에서 Value quantile 계산
rc.add_axis('value', cs_quantile(rc.data.btm, bins=3, labels=['low', 'mid', 'high'],
                                   group_by='size'))
# group_by='size'는 rc.rules['size']를 참조하여
# 'small' 그룹과 'big' 그룹 내에서 각각 독립적으로 quantile을 계산

# 3. HML 포트폴리오 구성 (각 Size 그룹 내에서 High-Low)
rc.init_signal_canvas('hml')
rc[rc.axis.value['high']] = 1.0   # Long high B/M (value) in both size groups
rc[rc.axis.value['low']] = -1.0   # Short low B/M (growth) in both size groups

# 4. 팩터 수익률 추적
hml_returns = rc.trace_pnl('hml')
print(f"HML Sharpe: {hml_returns['sharpe']:.2f}")
```

#### C. 로우레벨 마스크 활용 (Advanced Custom Logic)

```python
# 유동성 필터링된 유니버스에서 모멘텀 팩터 구성
rc.add_data('volume', Field('volume'))
rc.add_data('returns', Field('returns'))

# 1. 고유동성 마스크 생성
high_liquidity = rc.data.volume > rc.data.volume.quantile(0.5)

# 2. 마스크 적용된 quantile 계산
rc.add_axis('momentum', cs_quantile(rc.data.returns, bins=5, labels=['q1','q2','q3','q4','q5'],
                                      mask=high_liquidity))
# mask=False인 종목은 NaN으로 처리됨

# 3. 롱-숏 포트폴리오
rc.init_signal_canvas('momentum_factor')
rc[rc.axis.momentum['q5']] = 1.0
rc[rc.axis.momentum['q1']] = -1.0

mom_returns = rc.trace_pnl('momentum_factor')
```

#### D. 다차원 팩터 조합 (Multi-Factor Strategy)

```python
# 독립 정렬로 3개 팩터 축 생성
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=3, labels=['small','mid','big']))
rc.add_axis('momentum', cs_quantile(rc.data.mom, bins=5, labels=['q1','q2','q3','q4','q5']))
rc.add_axis('quality', cs_quantile(rc.data.roe, bins=3, labels=['low','mid','high']))

# 복잡한 다차원 선택
rc.init_signal_canvas('multi_factor')

# Small & High Momentum & High Quality
long_mask = (rc.axis.size['small'] & 
             rc.axis.momentum['q5'] & 
             rc.axis.quality['high'])

# Big & Low Momentum & Low Quality
short_mask = (rc.axis.size['big'] & 
              rc.axis.momentum['q1'] & 
              rc.axis.quality['low'])

rc[long_mask] = 1.0
rc[short_mask] = -1.0

multi_returns = rc.trace_pnl('multi_factor')
```

**설계 의도:**

- 이 패턴들이 alpha-canvas의 **핵심 활용 사례**입니다.
- **Fama-French 재현**: `group_by`로 종속 정렬을 간결하게 표현
- **유연성**: `mask`로 커스텀 유니버스 필터링 가능
- 듀얼 인터페이스(Formula + Selector)의 조합으로 복잡한 다차원 팩터 포트폴리오를 간결하게 표현합니다.
- 레이블 기반 선택(`'small'`, `'q5'`, `'high'`)으로 가독성과 유지보수성을 확보합니다.

## 3.3. 연산자 구현 패턴 (Operator Implementation Pattern)

### 3.3.1. 책임 분리 원칙

**핵심 원칙:** 연산자는 자신의 계산 로직을 소유하고, Visitor는 순회 및 캐싱만 담당합니다.

**잘못된 패턴 (Anti-Pattern):**

```python
# ❌ BAD: Visitor가 계산 로직을 포함
class EvaluateVisitor:
    def visit_ts_mean(self, node):
        child_result = node.child.accept(self)
        # Visitor 안에 rolling 계산 로직이 들어감 (잘못됨!)
        result = child_result.rolling(time=node.window, min_periods=node.window).mean()
        self._cache_result("TsMean", result)
        return result
```

**올바른 패턴 (Correct Pattern):**

```python
# ✅ GOOD: 연산자가 계산 로직을 소유
@dataclass
class TsMean(Expression):
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Visitor 인터페이스: 순회를 위한 진입점"""
        return visitor.visit_ts_mean(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """핵심 계산 로직: 연산자가 소유"""
        return child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).mean()

# ✅ GOOD: Visitor는 순회 및 캐싱만 담당
class EvaluateVisitor:
    def visit_ts_mean(self, node: TsMean) -> xr.DataArray:
        """트리 순회 및 상태 수집"""
        # 1. 순회: 자식 노드 평가
        child_result = node.child.accept(self)
        
        # 2. 계산 위임: 연산자에게 맡김
        result = node.compute(child_result)
        
        # 3. 상태 수집: 결과 캐싱
        self._cache_result("TsMean", result)
        
        return result
```

### 3.3.2. 연산자 구현 체크리스트

모든 연산자는 다음 구조를 따라야 합니다:

```python
@dataclass
class OperatorName(Expression):
    """연산자 설명.
    
    Args:
        child: 자식 Expression (필요시)
        param1: 연산자 파라미터 1
        param2: 연산자 파라미터 2
    
    Returns:
        연산 결과 DataArray
    """
    child: Expression  # 자식 노드 (있는 경우)
    param1: type1      # 연산자 파라미터들
    param2: type2
    
    def accept(self, visitor) -> xr.DataArray:
        """Visitor 인터페이스."""
        return visitor.visit_operator_name(self)
    
    def compute(self, *inputs: xr.DataArray) -> xr.DataArray:
        """핵심 계산 로직.
        
        Args:
            *inputs: 자식 노드들의 평가 결과
        
        Returns:
            이 연산의 결과 DataArray
        
        Note:
            이 메서드는 순수 함수여야 합니다 (부작용 없음).
            Visitor 참조 없이 독립적으로 테스트 가능해야 합니다.
        """
        # 실제 계산 로직
        result = ...  # xarray/numpy 연산
        return result
```

**체크리스트:**

- [ ] `accept()` 메서드: Visitor 인터페이스 제공
- [ ] `compute()` 메서드: 핵심 계산 로직 캡슐화
- [ ] `compute()`는 순수 함수 (부작용 없음)
- [ ] `compute()`는 Visitor 독립적 (직접 테스트 가능)
- [ ] Docstring으로 Args/Returns 명확히 문서화

### 3.3.3. Visitor 구현 패턴

모든 `visit_*()` 메서드는 동일한 3단계 패턴을 따릅니다:

```python
def visit_operator_name(self, node: OperatorName) -> xr.DataArray:
    """연산자 노드 방문: 순회 및 캐싱.
    
    Args:
        node: 연산자 Expression 노드
    
    Returns:
        연산 결과 DataArray
    """
    # 1️⃣ 순회(Traversal): 자식 노드들 평가
    child_result_1 = node.child1.accept(self)  # 깊이 우선
    child_result_2 = node.child2.accept(self)  # (있는 경우)
    
    # 2️⃣ 계산 위임(Delegation): 연산자에게 맡김
    result = node.compute(child_result_1, child_result_2)
    
    # 3️⃣ 상태 수집(State Collection): 결과 캐싱
    self._cache_result("OperatorName", result)
    
    return result
```

**Visitor의 역할:**

- ✅ **트리 순회:** 깊이 우선으로 자식 노드 방문
- ✅ **계산 위임:** `node.compute()`로 계산 맡김
- ✅ **상태 수집:** 중간 결과를 정수 스텝으로 캐싱
- ❌ **계산 로직 포함 금지:** rolling, rank, quantile 등의 로직은 연산자에 속함

### 3.3.4. 테스트 전략

**1. 연산자 단위 테스트 (Operator Unit Tests):**

```python
def test_ts_mean_compute_directly():
    """TsMean.compute() 메서드를 직접 테스트 (Visitor 없이)."""
    # 입력 데이터 준비
    data = xr.DataArray(
        [[1, 2], [3, 4], [5, 6]],
        dims=['time', 'asset']
    )
    
    # 연산자 생성
    operator = TsMean(child=Field('dummy'), window=2)
    
    # compute() 직접 호출 (Visitor 우회)
    result = operator.compute(data)
    
    # 검증
    assert np.isnan(result.values[0, 0])  # 첫 행 NaN
    assert result.values[1, 0] == 2.0     # mean([1, 3])
```

**2. 통합 테스트 (Integration Tests):**

```python
def test_ts_mean_with_visitor():
    """TsMean이 Visitor와 통합되어 작동하는지 테스트."""
    ds = xr.Dataset({'returns': data})
    visitor = EvaluateVisitor(ds)
    
    expr = TsMean(child=Field('returns'), window=3)
    result = visitor.evaluate(expr)
    
    # 캐싱 검증
    assert len(visitor._cache) == 2  # Field + TsMean
```

### 3.3.5. 이점 요약

| 측면 | 잘못된 패턴 | 올바른 패턴 |
|------|-------------|-------------|
| **책임** | Visitor가 모든 계산 담당 | 연산자가 자신의 계산 소유 |
| **테스트** | Visitor를 통해서만 테스트 | `compute()` 직접 테스트 가능 |
| **유지보수** | Visitor가 비대해짐 | 각 연산자 독립적 |
| **확장성** | 새 연산자마다 Visitor 수정 | Visitor 수정 최소화 |
| **단일 책임** | Visitor가 다중 책임 | 각 클래스 단일 책임 |

---

## 3.4. Cross-Sectional Quantile 연산자 구현 ✅ **IMPLEMENTED**

### 3.4.1. `CsQuantile` Expression 클래스 (실제 구현)

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import xarray as xr

@dataclass(eq=False)  # eq=False to preserve Expression comparison operators
class CsQuantile(Expression):
    """Cross-sectional quantile bucketing - returns categorical labels.
    
    Preserves input (T, N) shape. Each timestep is independently bucketed.
    Supports both independent sort (whole universe) and dependent sort
    (within groups via group_by parameter).
    """
    child: Expression  # 버킷화할 데이터 (e.g., Field('market_cap'))
    bins: int  # 버킷 개수
    labels: List[str]  # 레이블 리스트 (길이 = bins)
    group_by: Optional[str] = None  # 종속 정렬용: field 이름 (string)
    
    def __post_init__(self):
        """Validate parameters."""
        if len(self.labels) != self.bins:
            raise ValueError(
                f"labels length ({len(self.labels)}) must equal bins ({self.bins})"
            )
    
    def accept(self, visitor):
        """Visitor 인터페이스."""
        return visitor.visit_operator(self)
    
    def compute(
        self, 
        child_result: xr.DataArray, 
        group_labels: Optional[xr.DataArray] = None
    ) -> xr.DataArray:
        """Apply quantile bucketing - 핵심 계산 로직."""
        if group_labels is None:
            return self._quantile_independent(child_result)
        else:
            return self._quantile_grouped(child_result, group_labels)
```

### 3.4.2. 독립 정렬 (Independent Sort) 구현

**핵심 패턴:** `xarray.groupby('time').map()` + `pd.qcut` + **flatten-reshape**

```python
def _quantile_independent(self, data: xr.DataArray) -> xr.DataArray:
    """Independent sort - qcut at each timestep across all assets.
    
    핵심: pd.qcut은 1D 입력이 필요하므로 flatten → qcut → reshape 패턴 사용
    """
    def qcut_at_timestep(data_slice):
        """Apply pd.qcut to a single timestep's cross-section."""
        try:
            # CRITICAL: Flatten to 1D for pd.qcut
            values_1d = data_slice.values.flatten()
            result = pd.qcut(
                values_1d, 
                q=self.bins, 
                labels=self.labels, 
                duplicates='drop'  # Handle edge cases gracefully
            )
            # CRITICAL: Reshape back to original shape
            result_array = np.array(result).reshape(data_slice.shape)
            return xr.DataArray(
                result_array, 
                dims=data_slice.dims, 
                coords=data_slice.coords
            )
        except Exception:
            # Edge case: all same values, all NaN, etc.
            return xr.DataArray(
                np.full_like(data_slice.values, np.nan, dtype=object),
                dims=data_slice.dims, 
                coords=data_slice.coords
            )
    
    # xarray.groupby('time').map() automatically concatenates back to (T, N)
    result = data.groupby('time').map(qcut_at_timestep)
    return result
```

### 3.4.3. 종속 정렬 (Dependent Sort) 구현

**핵심 패턴:** 중첩된 groupby (groups → time → qcut)

```python
def _quantile_grouped(
    self, 
    data: xr.DataArray, 
    groups: xr.DataArray
) -> xr.DataArray:
    """Dependent sort - qcut within each group at each timestep.
    
    Nested groupby pattern:
    1. Group by categorical labels (e.g., 'small', 'big')
    2. Within each group, apply independent sort (group by time → qcut)
    3. xarray automatically concatenates results back to (T, N) shape
    """
    def apply_qcut_within_group(group_data: xr.DataArray) -> xr.DataArray:
        """Apply qcut at each timestep within this group."""
        return self._quantile_independent(group_data)
    
    # Nested groupby: groups → time → qcut
    # xarray automatically concatenates results back
    result = data.groupby(groups).map(apply_qcut_within_group)
    return result
```

### 3.4.4. Visitor 통합 (Special Case Handling)

**CsQuantile은 `visit_operator()`에서 특별 처리 필요 (group_by 조회)**:

```python
# In EvaluateVisitor.visit_operator()
from alpha_canvas.ops.classification import CsQuantile

# Special handling for CsQuantile (needs group_by lookup)
if isinstance(node, CsQuantile):
    # 1. Evaluate child
    child_result = node.child.accept(self)
    
    # 2. Look up group_by field if specified
    group_labels = None
    if node.group_by is not None:
        if node.group_by not in self._data:
            raise ValueError(
                f"group_by field '{node.group_by}' not found in dataset"
            )
        group_labels = self._data[node.group_by]
    
    # 3. Delegate to compute()
    result = node.compute(child_result, group_labels)
    
    # 4. Apply universe masking (automatic)
    if self._universe_mask is not None:
        result = result.where(self._universe_mask, np.nan)
    
    # 5. Cache
    self._cache_result("CsQuantile", result)
    return result
```

### 3.4.5. 핵심 구현 교훈 (실험에서 발견)

**1. Flatten-Reshape 패턴 필수:**
- `pd.qcut`은 1D 배열만 받음
- `data_slice.values.flatten()` → qcut → `reshape(data_slice.shape)`
- 이 패턴 없이는 shape 보존 불가능

**2. xarray.groupby().map() vs .apply():**
- `.map()`이 xarray → xarray 변환에 더 깔끔
- 자동 concatenation으로 shape 보존
- `.apply()`도 작동하지만 pandas 반환 시 사용

**3. duplicates='drop' 필수:**
- 모든 값이 동일한 edge case 처리
- 모든 NaN인 경우 graceful degradation
- 에러 발생 대신 NaN 반환

**4. 종속 정렬 성능:**
- 독립 정렬: ~27ms for (10, 6) data
- 종속 정렬: ~117ms for (10, 6) data (4.26x overhead)
- **허용 가능:** 팩터 연구는 배치 처리 (실시간 아님)

**5. 검증 방법:**
- 독립 vs 종속 정렬의 cutoff가 **달라야 함**
- 실험에서 17%의 positions가 다른 label 받음
- Fama-French 논문 methodology와 일치

### 3.4.6. 사용 예시 (실제 코드)

```python
from alpha_canvas.ops.classification import CsQuantile
from alpha_canvas.core.expression import Field

# 독립 정렬: 전체 유니버스에서 quantile
size_expr = CsQuantile(
    child=Field('market_cap'),
    bins=2,
    labels=['small', 'big']
)

# 종속 정렬: size 그룹 내에서 value quantile (Fama-French)
value_expr = CsQuantile(
    child=Field('book_to_market'),
    bins=3,
    labels=['low', 'mid', 'high'],
    group_by='size'  # 'size' field를 먼저 조회 → 각 그룹별 quantile
)

# 사용
rc.add_data('size', size_expr)  # 먼저 size 생성
rc.add_data('value', value_expr)  # size 그룹 내에서 value 계산

# Boolean Expression 통합
small_value = (rc.data['size'] == 'small') & (rc.data['value'] == 'high')
```

## 3.4. Property Accessor 구현 ✅ **IMPLEMENTED**

```python
from alpha_canvas.core.expression import Field


class DataAccessor:
    """Returns Field Expressions for lazy evaluation."""
    
    def __getitem__(self, field_name: str) -> Field:
        """Return Field Expression (not raw data!)"""
        if not isinstance(field_name, str):
            raise TypeError(
                f"Field name must be string, got {type(field_name).__name__}"
            )
        return Field(field_name)
    
    def __getattr__(self, name: str):
        """Prevent attribute access - item access only."""
        raise AttributeError(
            f"DataAccessor does not support attribute access. "
            f"Use rc.data['{name}'] instead of rc.data.{name}"
        )


# AlphaCanvas 통합
class AlphaCanvas:
    def __init__(self, ...):
        self._data_accessor = DataAccessor()
    
    @property
    def data(self) -> DataAccessor:
        """Access data fields as Field Expressions."""
        return self._data_accessor
```

**사용 예시**:

```python
# Basic field access
field = rc.data['size']  # Returns Field('size')

# Comparison creates Expression
mask = rc.data['size'] == 'small'  # Returns Equals Expression

# Evaluate
result = rc.evaluate(mask)  # Boolean DataArray with universe masking
```

---

## 3.4.2. Signal Assignment (Lazy Evaluation) ✅ **IMPLEMENTED**

### 개요

**Signal Assignment**는 Expression 객체에 값을 할당하여 시그널을 구성하는 기능입니다. Fama-French 팩터와 같은 복잡한 시그널을 직관적인 문법으로 생성할 수 있습니다.

**핵심 설계 원칙**:
- **Lazy Evaluation**: 할당은 저장만 하고 즉시 실행하지 않음
- **Implicit Canvas**: 별도의 캔버스 생성 없이 Expression 결과가 캔버스 역할
- **Traceability**: Base result와 final result를 별도로 캐싱하여 추적 가능
- **DRY Principle**: Lazy initialization으로 모든 Expression에서 자동 작동

### Expression.__setitem__ 구현 (DRY Lazy Initialization)

```python
class Expression(ABC):
    """Base class for all Expressions.
    
    Supports lazy assignment via __setitem__:
        signal[mask] = value  # Stores assignment, does not evaluate
    """
    
    def __setitem__(self, mask, value):
        """Store assignment for lazy evaluation.
        
        Uses lazy initialization - _assignments list is created on first use.
        This follows the DRY principle: no __post_init__ needed in subclasses.
        
        Args:
            mask: Boolean Expression or DataArray indicating where to assign
            value: Scalar value to assign where mask is True
        
        Note:
            Assignments are stored as (mask, value) tuples and applied sequentially
            during evaluation. Later assignments overwrite earlier ones for overlapping
            positions.
        """
        # Lazy initialization - create _assignments if it doesn't exist
        if not hasattr(self, '_assignments'):
            self._assignments = []
        
        self._assignments.append((mask, value))
```

**Lazy Initialization의 장점**:
1. ✅ **No Boilerplate**: 모든 Expression 서브클래스에 `__post_init__` 불필요
2. ✅ **DRY Principle**: 중복 코드 제거
3. ✅ **Automatic**: 모든 Expression에서 자동으로 작동
4. ✅ **Efficient**: 할당이 없으면 `_assignments` 속성도 생성되지 않음

### Visitor Integration

```python
class EvaluateVisitor:
    def evaluate(self, expr: Expression) -> xr.DataArray:
        """Evaluate expression and apply assignments if present."""
        # Step 1: Evaluate base expression (tree traversal)
        base_result = expr.accept(self)
        
        # Step 2: Check if expression has assignments (lazy initialization)
        assignments = getattr(expr, '_assignments', None)
        if assignments:
            # Cache base result for traceability
            base_name = f"{expr.__class__.__name__}_base"
            self._cache[self._step_counter] = (base_name, base_result)
            self._step_counter += 1
            
            # Apply assignments sequentially
            final_result = self._apply_assignments(base_result, assignments)
            
            # Apply universe masking to final result
            if self._universe_mask is not None:
                final_result = final_result.where(self._universe_mask)
            
            # Cache final result
            final_name = f"{expr.__class__.__name__}_with_assignments"
            self._cache[self._step_counter] = (final_name, final_result)
            self._step_counter += 1
            
            return final_result
        
        # No assignments, return base result as-is
        return base_result
    
    def _apply_assignments(self, base_result: xr.DataArray, assignments: list) -> xr.DataArray:
        """Apply assignments sequentially to base result."""
        result = base_result.copy(deep=True)
        
        for mask_expr, value in assignments:
            # If mask is an Expression, evaluate it
            if hasattr(mask_expr, 'accept'):
                mask_data = mask_expr.accept(self)
            else:
                # Already a DataArray or numpy array
                mask_data = mask_expr
            
            # Ensure mask is boolean (required for ~ operator)
            mask_bool = mask_data.astype(bool)
            
            # Apply assignment: replace values where mask is True
            result = result.where(~mask_bool, value)
        
        return result
```

### Constant Expression (Blank Canvas)

```python
from dataclasses import dataclass
import numpy as np
import xarray as xr
from alpha_canvas.core.expression import Expression


@dataclass(eq=False)
class Constant(Expression):
    """Expression that produces a constant-valued DataArray.
    
    Creates a universe-shaped (T, N) DataArray filled with the specified
    constant value. Serves as a "blank canvas" for signal construction.
    
    Example:
        >>> signal = Constant(0.0)  # Blank canvas (all zeros)
        >>> signal[mask1] = 1.0     # Assign long positions
        >>> signal[mask2] = -1.0    # Assign short positions
    """
    value: float
    
    def accept(self, visitor):
        return visitor.visit_constant(self)
```

### 사용 예시: Fama-French 2×3 Factor

```python
# Step 1: Create size and value classifications
rc.add_data('size', CsQuantile(Field('market_cap'), bins=2, labels=['small', 'big']))
rc.add_data('value', CsQuantile(Field('book_to_market'), bins=3, labels=['low', 'medium', 'high']))

# Step 2: Create selector masks
is_small = rc.data['size'] == 'small'
is_big = rc.data['size'] == 'big'
is_low = rc.data['value'] == 'low'
is_high = rc.data['value'] == 'high'

# Step 3: Construct signal with lazy assignments
signal = Constant(0.0)                # Implicit blank canvas
signal[is_small & is_high] = 1.0      # Small/High-Value (long)
signal[is_big & is_low] = -1.0        # Big/Low-Value (short)

# Step 4: Evaluate (assignments applied here)
result = rc.evaluate(signal)

# Result: universe-shaped (T, N) array
#  - 1.0 where size=='small' AND value=='high'
#  - -1.0 where size=='big' AND value=='low'
#  - 0.0 elsewhere (neutral)
#  - NaN outside universe
```

### Overlapping Masks (Sequential Application)

```python
signal = Constant(0.0)
signal[is_small] = 0.5                # All small caps = 0.5
signal[is_small & is_high] = 1.0      # Small/High overwrites to 1.0

# Result: Later assignment wins for overlapping positions
#  - Small/High: 1.0 (overwritten)
#  - Small/Other: 0.5 (from first assignment)
#  - Others: 0.0 (from Constant)
```

### Traceability

```python
# After evaluation, check cached steps
for step_idx in sorted(rc._evaluator._cache.keys()):
    name, data = rc._evaluator._cache[step_idx]
    print(f"Step {step_idx}: {name}")

# Output:
#   Step 0: Constant_0.0_base             # Base constant array
#   Step 1: Field_size                     # Size classification
#   Step 2: Field_value                    # Value classification
#   Step 3: Equals                         # is_small mask
#   Step 4: Equals                         # is_high mask
#   Step 5: And                            # is_small & is_high
#   Step 6: Constant_0.0_with_assignments  # Final signal with assignments
```

**핵심 장점**:
- Base result와 final result가 별도 단계로 캐싱됨
- PnL tracking에 필수적인 기능
- 각 할당의 영향을 단계별로 추적 가능

### Implementation Checklist

- ✅ `Expression.__setitem__` with lazy initialization (DRY)
- ✅ `Visitor.evaluate()` handles assignments
- ✅ `Visitor._apply_assignments()` sequential application
- ✅ `Constant` Expression for blank canvas
- ✅ `visit_constant()` in Visitor
- ✅ Boolean mask conversion (`.astype(bool)`)
- ✅ Universe masking integration
- ✅ Traceability (separate base/final caching)
- ✅ Tests: storage, evaluation, overlapping masks, caching
- ✅ Showcase: Fama-French 2×3 factor construction

---

## 3.4.3. Portfolio Weight Scaling 📋 **PLANNED**

### 개요

**Portfolio Weight Scaling**은 임의의 시그널 값을 제약 조건을 만족하는 포트폴리오 가중치로 변환하는 모듈입니다. Strategy Pattern을 사용하여 다양한 스케일링 전략을 플러그인 방식으로 지원합니다.

**핵심 설계 원칙**:
- **Stateless**: 스케일러는 상태를 저장하지 않음 (항상 명시적 파라미터로 전달)
- **Strategy Pattern**: 다양한 스케일링 전략을 쉽게 교체 가능
- **Cross-Sectional**: 각 시점 독립적으로 처리 (`groupby('time').map()` 패턴)
- **NaN-Aware**: 유니버스 마스킹 자동 보존

### WeightScaler 베이스 클래스

```python
from abc import ABC, abstractmethod
import xarray as xr


class WeightScaler(ABC):
    """Abstract base class for weight scaling strategies.
    
    Converts arbitrary signal values to portfolio weights by applying
    constraints cross-sectionally (independently for each time period).
    
    Philosophy:
    - Operates on (T, N) signal DataArray
    - Returns (T, N) weight DataArray
    - Each time slice processed independently
    - NaN-aware (respects universe masking)
    - Strategy pattern: subclasses define scaling logic
    """
    
    @abstractmethod
    def scale(self, signal: xr.DataArray) -> xr.DataArray:
        """Scale signal to weights.
        
        Args:
            signal: (T, N) DataArray with arbitrary signal values
        
        Returns:
            (T, N) DataArray with portfolio weights
        
        Note:
            - Must handle NaN values (preserve them in output)
            - Must process each time slice independently
            - Should validate inputs (not all NaN, etc.)
        """
        pass
    
    def _validate_signal(self, signal: xr.DataArray):
        """Validate signal before scaling."""
        if signal.dims != ('time', 'asset'):
            raise ValueError(
                f"Signal must have dims ('time', 'asset'), got {signal.dims}"
            )
        
        # Check if any time slice has non-NaN values
        non_nan_counts = (~signal.isnull()).sum(dim='asset')
        if (non_nan_counts == 0).all():
            raise ValueError(
                "All signal values are NaN across all time periods"
            )
```

### GrossNetScaler (통합 프레임워크)

```python
class GrossNetScaler(WeightScaler):
    """Unified weight scaler based on gross and net exposure targets.
    
    Uses the unified framework:
        L_target = (G + N) / 2
        S_target = (G - N) / 2
    
    Where:
        G = target_gross_exposure = sum(abs(weights))
        N = target_net_exposure = sum(weights)
        L = sum of positive weights
        S = sum of negative weights (negative value)
    
    Args:
        target_gross: Target gross exposure (default: 2.0 for 200% gross)
        target_net: Target net exposure (default: 0.0 for dollar-neutral)
    
    Example:
        >>> # Dollar neutral: L=1.0, S=-1.0
        >>> scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        >>> 
        >>> # Net long 10%: L=1.1, S=-0.9
        >>> scaler = GrossNetScaler(target_gross=2.0, target_net=0.2)
        >>> 
        >>> # Crypto futures: L=0.5, S=-0.5
        >>> scaler = GrossNetScaler(target_gross=1.0, target_net=0.0)
    """
    
    def __init__(self, target_gross: float = 2.0, target_net: float = 0.0):
        self.target_gross = target_gross
        self.target_net = target_net
        
        # Validate constraints
        if target_gross < 0:
            raise ValueError("target_gross must be non-negative")
        if abs(target_net) > target_gross:
            raise ValueError(
                "Absolute net exposure cannot exceed gross exposure"
            )
        
        # Calculate target long and short books
        self.L_target = (target_gross + target_net) / 2.0
        self.S_target = (target_net - target_gross) / 2.0  # Negative value
    
    def scale(self, signal: xr.DataArray) -> xr.DataArray:
        """Scale signal using fully vectorized gross/net exposure constraints.
        
        Key innovation: NO ITERATION - pure vectorized operations.
        Always meets gross target, even for one-sided signals.
        """
        self._validate_signal(signal)
        
        # Step 1: Separate positive/negative (vectorized)
        s_pos = signal.where(signal > 0, 0.0)
        s_neg = signal.where(signal < 0, 0.0)
        
        # Step 2: Sum along asset dimension (vectorized)
        sum_pos = s_pos.sum(dim='asset', skipna=True)  # Shape: (time,)
        sum_neg = s_neg.sum(dim='asset', skipna=True)  # Shape: (time,)
        
        # Step 3: Normalize (vectorized, handles 0/0 → nan → 0)
        norm_pos = (s_pos / sum_pos).fillna(0.0)
        norm_neg_abs = (np.abs(s_neg) / np.abs(sum_neg)).fillna(0.0)
        
        # Step 4: Apply L/S targets (vectorized)
        weights_long = norm_pos * self.L_target
        weights_short_mag = norm_neg_abs * np.abs(self.S_target)
        
        # Step 5: Combine (subtract to make short side negative)
        weights = weights_long - weights_short_mag
        
        # Step 6: Calculate actual gross per row (vectorized)
        actual_gross = np.abs(weights).sum(dim='asset', skipna=True)  # Shape: (time,)
        
        # Step 7: Scale to meet target gross (vectorized)
        # Use xr.where to avoid inf from 0/0
        scale_factor = xr.where(actual_gross > 0, self.target_gross / actual_gross, 1.0)
        final_weights = weights * scale_factor
        
        # Step 8: Convert computational NaN to 0 (BEFORE universe mask)
        final_weights = final_weights.fillna(0.0)
        
        # Step 9: Apply universe mask (preserves NaN where signal was NaN)
        final_weights = final_weights.where(~signal.isnull())
        
        return final_weights
```

### 편의 Scaler 클래스들

```python
class DollarNeutralScaler(GrossNetScaler):
    """Dollar neutral: sum(long) = 1.0, sum(short) = -1.0.
    
    Convenience wrapper for GrossNetScaler(2.0, 0.0).
    
    This is the most common scaler for market-neutral strategies.
    """
    def __init__(self):
        super().__init__(target_gross=2.0, target_net=0.0)


class LongOnlyScaler(WeightScaler):
    """Long-only portfolio: sum(weights) = target_long.
    
    Ignores negative signals, normalizes positive signals to sum to target.
    Uses vectorized operations for efficiency.
    
    Args:
        target_long: Target sum of weights (default: 1.0)
    
    Example:
        >>> scaler = LongOnlyScaler(target_long=1.0)
        >>> # All negative signals become 0, positives sum to 1.0
    """
    
    def __init__(self, target_long: float = 1.0):
        self.target_long = target_long
    
    def scale(self, signal: xr.DataArray) -> xr.DataArray:
        """Scale using vectorized long-only normalization."""
        self._validate_signal(signal)
        
        # Only keep positive values (vectorized)
        s_pos = signal.where(signal > 0, 0.0)
        
        # Sum along asset dimension (vectorized)
        sum_pos = s_pos.sum(dim='asset', skipna=True)  # Shape: (time,)
        
        # Normalize and scale (vectorized, handles 0/0 → nan → 0)
        weights = (s_pos / sum_pos * self.target_long).fillna(0.0)
        
        # Preserve NaN where signal was NaN (universe masking)
        return weights.where(~signal.isnull())
```

### Facade 통합

```python
# In AlphaCanvas class (src/alpha_canvas/core/facade.py)

def scale_weights(
    self, 
    signal: Union[Expression, xr.DataArray], 
    scaler: 'WeightScaler'
) -> xr.DataArray:
    """Scale signal to portfolio weights.
    
    Args:
        signal: Expression or DataArray with signal values
        scaler: WeightScaler strategy instance (REQUIRED)
    
    Returns:
        (T, N) DataArray with portfolio weights
    
    Note:
        Scaler is a required parameter - no default.
        This is intentional for explicit, research-friendly API.
    
    Example:
        >>> from alpha_canvas.portfolio import DollarNeutralScaler
        >>> 
        >>> signal = ts_mean(Field('returns'), 5)
        >>> scaler = DollarNeutralScaler()
        >>> weights = rc.scale_weights(signal, scaler)
        >>> 
        >>> # Compare multiple scalers
        >>> w1 = rc.scale_weights(signal, DollarNeutralScaler())
        >>> w2 = rc.scale_weights(signal, LongOnlyScaler(1.0))
    """
    # Evaluate if Expression
    if hasattr(signal, 'accept'):
        signal_data = self.evaluate(signal)
    else:
        signal_data = signal
    
    # Apply scaling strategy
    weights = scaler.scale(signal_data)
    
    return weights
```

### 사용 패턴 및 예시

**패턴 1: 직접 사용 (가장 명시적)**

```python
from alpha_canvas.portfolio import DollarNeutralScaler

# 1. Signal 생성
signal_expr = ts_mean(Field('returns'), 5)
signal_data = rc.evaluate(signal_expr)

# 2. Scaler 생성 및 적용
scaler = DollarNeutralScaler()
weights = scaler.scale(signal_data)

# 검증
assert abs(weights[weights > 0].sum() - 1.0) < 1e-6  # Long = 1.0
assert abs(weights[weights < 0].sum() + 1.0) < 1e-6  # Short = -1.0
```

**패턴 2: Facade 편의 메서드**

```python
from alpha_canvas.portfolio import GrossNetScaler

signal_expr = ts_mean(Field('returns'), 5)
scaler = GrossNetScaler(target_gross=2.0, target_net=0.2)

# evaluate + scale 한 번에
weights = rc.scale_weights(signal_expr, scaler)
```

**패턴 3: 여러 스케일러 비교 (연구용)**

```python
from alpha_canvas.portfolio import (
    DollarNeutralScaler,
    GrossNetScaler,
    LongOnlyScaler
)

# 동일 시그널에 여러 스케일링 전략 적용
signal = rc.evaluate(my_alpha_expr)

strategies = {
    'dollar_neutral': DollarNeutralScaler(),
    'net_long_10pct': GrossNetScaler(2.0, 0.2),
    'long_only': LongOnlyScaler(1.0),
    'crypto_futures': GrossNetScaler(1.0, 0.0)
}

weights_dict = {
    name: scaler.scale(signal)
    for name, scaler in strategies.items()
}

# 각 전략 비교
for name, weights in weights_dict.items():
    print(f"{name}:")
    print(f"  Gross: {abs(weights).sum()}")
    print(f"  Net: {weights.sum()}")
```

### Module Structure

```
src/alpha_canvas/portfolio/
├── __init__.py              # Export all scalers
├── base.py                  # WeightScaler abstract base class
└── strategies.py            # GrossNetScaler, DollarNeutralScaler, LongOnlyScaler
```

### 테스트 전략

**1. 단위 테스트** (`tests/test_portfolio/test_strategies.py`):
- `GrossNetScaler` 수학 검증 (L_target, S_target 계산)
- 각 scaler의 constraint 충족 확인
- NaN 보존 검증
- Edge cases: all positive, all negative, zeros

**2. 통합 테스트**:
- AlphaCanvas.scale_weights() 통합
- Expression → evaluation → scaling 전체 파이프라인
- Universe masking 보존 검증

**3. 성능 테스트**:
- 크로스-섹션 독립성 검증
- Large dataset (T=1000, N=3000) 벤치마크

### Implementation Checklist

- [ ] `WeightScaler` abstract base class
- [ ] `GrossNetScaler` with unified framework (fully vectorized)
- [ ] `DollarNeutralScaler` convenience wrapper
- [ ] `LongOnlyScaler` implementation (fully vectorized)
- [ ] `AlphaCanvas.scale_weights()` facade method
- [ ] Unit tests for each scaler
- [ ] Integration tests with facade
- [x] **Experiment: weight scaling validation** (exp_18_weight_scaling.py - ALL PASS ✅)
- [ ] Showcase: Fama-French signal → weights
- [x] **Documentation: FINDINGS.md updated** ✅
- [x] **Documentation: architecture.md updated** ✅
- [x] **Documentation: implementation.md updated** ✅

**Performance Validated**:
- Small (10×6): 7ms
- Medium (100×50): 7ms
- 1Y Daily (252×100): 8ms
- Large (1000×500): 34ms
- **10x-220x speedup** vs iterative approach

---

## 3.5. 개발 원칙

### 3.5.1. 지연 평가 (Lazy Evaluation)

- `Expression` 객체는 "레시피"이며 데이터를 가지지 않습니다.
- `EvaluateVisitor`가 실제 평가를 담당하며, 이때 캐싱이 발생합니다.
- 불필요한 재계산을 방지하여 성능을 최적화합니다.

### 3.5.2. 레이블 우선 (Label-first)

- 모든 버킷 연산은 정수 인덱스 대신 **의미 있는 레이블**을 반환해야 합니다.
- 예: `cs_quantile(..., labels=['small', 'mid', 'big'])`
- 이는 PRD의 핵심 문제 2를 해결하는 설계 원칙입니다.

### 3.5.3. 추적성 우선 (Traceability-first)

- 모든 중간 계산 결과는 **정수 step 인덱스**로 캐시되어야 합니다.
- 사용자는 재계산 없이 모든 중간 단계를 검사할 수 있어야 합니다.
- 이는 PRD의 핵심 문제 3을 해결하는 설계 원칙입니다.

### 3.5.4. Pythonic 우선 (Pythonic-first)

- 문자열 DSL 대신 Python의 네이티브 문법을 활용합니다.
- 예: `&`, `|`, `[]`, `=` 연산자 오버로딩
- IDE 자동완성 및 타입 힌트를 최대한 활용합니다.

### 3.5.5. 종속 정렬 지원 (Dependent Sort Support)

- `cs_quantile`은 `group_by` 파라미터로 종속 정렬을 지원해야 합니다.
- 이는 Fama-French 팩터 재현을 위한 핵심 요구사항입니다.
- `mask` 파라미터로 로우레벨 커스터마이징도 가능해야 합니다.

## 3.6. 테스트 전략

### 3.6.1. 단위 테스트

- `Expression` 각 노드 클래스
- `EvaluateVisitor` 메서드별 테스트 (특히 `visit_cs_quantile`의 `group_by` 로직)
- `ConfigLoader` YAML 파싱 테스트
- 정수 step 인덱싱 로직

### 3.6.2. 통합 테스트

- 전체 워크플로우 (초기화 → 데이터 로드 → Expression 평가 → PnL 추적)
- 듀얼 인터페이스 조합 시나리오
- 독립/종속 이중 정렬 기반 팩터 수익률 계산 패턴
- Fama-French SMB, HML 팩터 재현 검증

### 3.6.3. 성능 테스트

- 대용량 데이터 (T=1000, N=3000) 처리 시간
- 캐싱 효과 검증 (step별 조회 성능)
- 종속 정렬의 오버헤드 측정
- 메모리 사용량 프로파일링

## 3.7. 코딩 컨벤션

- **타입 힌트:** 모든 public 메서드에 타입 힌트 필수
- **Docstring:** Google 스타일 docstring 사용
- **Linting:** `ruff` 사용
- **Formatting:** `black` 사용
- **Import 순서:** 표준 라이브러리 → 서드파티 → 로컬

## 3.8. 인터페이스 마이그레이션 가이드

### 3.8.1. Step 인덱싱 변경사항

**이전 (문자열 기반):**

```python
# ❌ 사용하지 마세요
rc.trace_pnl('alpha1', step='ts_mean')
rc.get_intermediate('alpha1', step='ts_mean')
```

**현재 (정수 기반):**

```python
# ✅ 올바른 사용법
rc.trace_pnl('alpha1', step=1)  # step 1까지 추적
rc.get_intermediate('alpha1', step=1)  # step 1 데이터 조회

# 모든 단계 추적
rc.trace_pnl('alpha1')  # step=None (기본값)
# 반환: {0: {...}, 1: {...}, 2: {...}}
```

### 3.8.2. 독립/종속 정렬 패턴

**독립 정렬 (변경 없음):**

```python
# ✅ 기존과 동일하게 작동
rc.add_axis('size', cs_quantile(rc.data.mcap, bins=2, labels=['small','big']))
```

**종속 정렬 (신규 기능):**

```python
# ✅ 새로운 기능
rc.add_axis('value', cs_quantile(rc.data.btm, bins=3, labels=['low','mid','high'],
                                   group_by='size'))
```

**마스크 기반 필터링 (신규 기능):**

```python
# ✅ 새로운 기능
mask = rc.data.volume > threshold
rc.add_axis('filtered', cs_quantile(rc.data.returns, bins=5, labels=[...],
                                      mask=mask))
```

## 3.9. 구현 성공 기준

### 3.9.1. Step 인덱싱 검증

✅ **필수 동작:**

- `rc.trace_pnl('alpha', step=2)` → step 2까지의 PnL 반환
- `rc.get_intermediate('alpha', step=2)` → step 2의 캐시된 DataArray 반환
- 병렬 Expression (브랜치가 있는 트리)에서 올바른 순서로 인덱싱
- 잘못된 step 인덱스 입력 시 명확한 에러 메시지

### 3.9.2. 종속 정렬 검증

✅ **필수 동작:**

- **독립 정렬**: `cs_quantile(...)` → 전체 유니버스 대상 quantile
- **종속 정렬**: `cs_quantile(..., group_by='axis')` → 각 그룹 내 quantile
- **마스크 필터링**: `cs_quantile(..., mask=...)` → 필터링된 부분집합 대상 quantile
- 독립/종속 정렬의 결과 cutoff가 명확히 다름 (검증 테스트 필요)

### 3.9.3. Fama-French 재현 검증

✅ **필수 동작:**

- SMB (독립 2×3 정렬) → 예상된 포트폴리오 가중치 생성
- HML (종속 2×3 정렬) → 예상된 포트폴리오 가중치 생성
- 독립/종속 방식의 cutoff 차이 검증 (academic paper 기준과 일치)

## 3.10. 다음 단계

### Phase 1: 핵심 컴포넌트 구현

- [ ] `Expression` 추상 클래스 및 Leaf/Composite 구현
- [ ] `EvaluateVisitor` 기본 구조 및 캐싱 메커니즘 (정수 step 카운터 포함)
- [ ] `ConfigLoader` 및 YAML 파싱
- [ ] `AlphaCanvas` Facade 기본 구조

### Phase 2: 연산자 구현

- [ ] Timeseries 연산자 (`ts_mean`, `ts_sum`, etc.)
- [ ] Cross-sectional 연산자 (`cs_rank`, `cs_quantile` with `group_by` and `mask`)
- [ ] Transform 연산자 (`group_neutralize`, etc.)

### Phase 3: 추적성 및 분석

- [ ] `PnLTracer` 구현
- [ ] 선택적 단계 추적 로직 (정수 인덱스 기반)
- [ ] 성과 지표 계산
- [ ] PnL 리포트에 step 메타데이터 표시

### Phase 4: 인터페이스 완성

- [ ] Property accessor (`rc.data`, `rc.axis`)
- [ ] NumPy-style 할당 (`rc[mask] = value`)
- [ ] 헬퍼 메서드 (`rc.ts_mean()` 등)

### Phase 5: 검증 및 테스트

- [ ] 정수 step 인덱싱 단위 테스트
- [ ] `_quantile_grouped` 로직 단위 테스트
- [ ] Fama-French SMB/HML 통합 테스트
- [ ] 종속 정렬 성능 벤치마크

---

**참고:** 이 문서는 실제 구현 과정에서 발견되는 새로운 패턴과 교훈을 지속적으로 반영해야 합니다 (Living Document).
