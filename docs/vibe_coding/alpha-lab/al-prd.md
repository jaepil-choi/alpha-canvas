# Alpha-Lab 제품 요구사항 문서 (Product Requirement Document)

## 1. 개요

**alpha-lab**은 alpha-canvas가 생성한 시그널 및 백테스트 결과를 분석하고 시각화하는 독립 패키지입니다.

### 핵심 원칙

* **Loosely Coupled**: alpha-canvas의 공개 API만 사용 (`get_signal()`, `get_weights()`, `get_port_return()`)
* **Jupyter-First**: 노트북 환경에 최적화된 인터랙티브 분석 및 시각화
* **Pluggable**: 확장 가능한 분석기 및 시각화 컴포넌트
* **No Internal Access**: Expression 트리나 Visitor 내부 접근 금지 (공개 API만 사용)

### 패키지 구조

```python
alpha_lab/
├── core/                    # 기반 클래스
│   ├── analyzer.py         # BaseAnalyzer (alpha-canvas 공개 API 소비)
│   └── metrics.py          # Metric 계산 함수들
├── performance/            # 성과 분석
│   ├── metrics.py          # PerformanceAnalyzer
│   ├── risk.py             # RiskAnalyzer
│   └── attribution.py      # Attribution 분석
├── visualization/          # 시각화
│   ├── base.py             # Visualizer 베이스
│   ├── heatmaps.py         # 2D 히트맵 (signal, weights, returns)
│   ├── curves.py           # PnL 곡선, underwater charts
│   └── themes.py           # 색상 스킴 (Korean vs US)
├── factor/                 # 팩터 분석
│   ├── exposure.py         # 팩터 노출 회귀
│   ├── ic.py               # Information Coefficient
│   └── turnover.py         # Turnover 분석
├── comparison/             # 비교 도구
│   ├── strategy.py         # 전략 비교
│   └── scaler.py           # Scaler 비교
└── advanced/               # 고급 기능 (미래)
    ├── sensitivity.py      # 민감도 분석
    ├── regime.py           # 체제 분석
    └── monte_carlo.py      # Monte Carlo 시뮬레이션
```

## 2. 핵심 기능 (MVP Scope)

### L1: 성과 지표 계산 (Performance Metrics)

**요구사항**: 백테스트 결과로부터 포괄적인 성과 지표를 계산합니다.

**MVP 지표**:
- **수익성**: Sharpe ratio, Sortino ratio, Calmar ratio, Total return
- **리스크**: Volatility (annualized), Max drawdown, Drawdown duration
- **거래**: Turnover (daily, annualized), Margin (PnL per dollar traded)
- **포지션**: Long/short stock counts (average), Position concentration
- **정확도**: Hit ratio (direction correctness: sign(weight) == sign(return))

**시간 집계**: Daily → Monthly → Yearly 변환 기능

**사용 예시**:

```python
from alpha_lab import PerformanceAnalyzer

# 분석기 초기화 (alpha-canvas 인스턴스 소비)
analyzer = PerformanceAnalyzer(rc)

# 단일 step 지표 계산
metrics = analyzer.compute_metrics(step=2)
print(f"Sharpe: {metrics['sharpe']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Turnover: {metrics['turnover_annual']:.2%}")

# Step 비교 (N vs N-1)
diff = analyzer.compare_steps(step_from=1, step_to=2)
print(f"Sharpe improvement: {diff['sharpe_delta']:+.2f}")

# Multi-step 진화
evolution = analyzer.step_evolution(steps=[0, 1, 2])
# Returns DataFrame: rows=steps, cols=metrics
```

### L2: PnL 추적 및 비교 (PnL Tracing)

**요구사항**: 단계별 PnL 변화를 추적하고 비교합니다.

**핵심 질문에 답변**:
- "이 연산자를 추가하니 성과가 개선되었나?"
- "어느 단계에서 PnL이 하락했나?"
- "최종 시그널과 원본 시그널의 차이는?"

**사용 예시**:

```python
# Step N vs Step N-1 비교
comparison = analyzer.compare_steps(1, 2)
print(f"Sharpe: {comparison['sharpe_before']:.2f} → {comparison['sharpe_after']:.2f}")
print(f"Change: {comparison['sharpe_delta']:+.2f}")

# 전체 step 진화 보기
df_evolution = analyzer.step_evolution(steps=[0, 1, 2])
#       Sharpe  MaxDD  Turnover
# Step0   0.5   -0.15    0.30
# Step1   0.7   -0.12    0.35
# Step2   0.6   -0.10    0.40
```

## 3. 시각화 기능 (Future Expansion)

### V1: 2D 히트맵 (Heatmaps)

**요구사항**: Signal, weights, returns를 2D (T × N) 히트맵으로 시각화합니다.

**특징**:
- **색상 스킴 선택 가능**:
  - Korean: 파란색(음수), 흰색(0), 빨간색(양수)
  - US: 빨간색(음수), 흰색(0), 초록색(양수)
- **인터랙티브**: Plotly 기반, zoom/pan/hover

```python
from alpha_lab import Visualizer

viz = Visualizer(rc, color_scheme='korean')

# Signal 히트맵
fig = viz.heatmap_signal(step=2)
fig.show()  # Jupyter 출력

# Weights 히트맵
fig = viz.heatmap_weights(step=2)

# Portfolio returns 히트맵
fig = viz.heatmap_returns(step=2)
```

### V2: PnL Evolution Curves

**요구사항**: 시간에 따른 PnL 변화를 라인 플롯으로 시각화합니다.

**Option B (Multi-step comparison)**: 여러 step의 누적 PnL을 겹쳐서 표시하여 연산자 추가 효과를 시각적으로 확인합니다.

```python
# 단일 step PnL 곡선
fig = viz.pnl_curve(step=2, cumulative=True)

# Multi-step 비교 (핵심!)
fig = viz.compare_pnl_curves(
    steps=[0, 1, 2],
    labels=['Raw Signal', 'After TsMean', 'After Rank']
)
# 3개 라인이 겹쳐져서 표시됨
# → 어느 연산자가 PnL을 개선/악화시켰는지 즉시 확인

# Underwater chart (drawdown 시각화)
fig = viz.underwater_chart(step=2)
```

### V3: Attribution Waterfall

**요구사항**: Winner/loser 기여도를 바 차트로 시각화합니다.

```python
# Top 10 winners, bottom 10 losers
fig = viz.attribution_waterfall(step=2, top_n=10)
# Green bars: winners, Red bars: losers
```

## 4. 고급 분석 기능 (Future Expansion)

### A1: Factor Exposure Analysis

**요구사항**: 시그널의 암묵적 팩터 노출을 분석합니다.

```python
from alpha_lab import FactorAnalyzer

factor_analyzer = FactorAnalyzer(rc)

# 팩터 회귀
exposures = factor_analyzer.regress_on_factors(
    step=2,
    factors=['market', 'size', 'value', 'momentum']
)
# Returns: betas, R², alpha

# Rolling factor betas
rolling_betas = factor_analyzer.rolling_betas(step=2, window='1Y')
```

### A2: IC (Information Coefficient) Analysis

**요구사항**: 시그널의 예측력을 분석합니다.

```python
from alpha_lab import ICAnalyzer

ic_analyzer = ICAnalyzer(rc)

# Signal predictiveness
ic = ic_analyzer.compute_ic(step=2, forward_returns=1)
# Returns: Pearson IC, Spearman Rank IC

# IC decay (signal persistence)
ic_decay = ic_analyzer.ic_decay(step=2, horizons=[1, 5, 10, 20])

# Rolling IC (stability)
rolling_ic = ic_analyzer.rolling_ic(step=2, window='3M')
```

### A3: Sensitivity Analysis (민감도 분석)

**요구사항**: 입력 데이터 변화에 대한 PnL 민감도를 분석합니다 ("diff" feature).

```python
from alpha_lab import SensitivityAnalyzer

sensitivity = SensitivityAnalyzer(rc)

# Field 데이터 perturbation
field_sensitivity = sensitivity.perturb_field(
    field_name='market_cap',
    perturbation=0.01,  # +1%
    step=2
)
# Returns: {'original_pnl': X, 'perturbed_pnl': Y, 'delta': Y-X}

# Intermediate signal perturbation
signal_sensitivity = sensitivity.perturb_signal(
    step=1,  # TsMean 결과 perturb
    perturbation=0.05,
    final_step=2
)

# Tornado chart (어느 입력이 가장 중요한가?)
fig = sensitivity.tornado_chart(
    fields=['market_cap', 'returns', 'volume'],
    perturbation_range=[-0.1, 0.1]
)
```

## 5. MVP 구현 우선순위

**Phase 1 (Immediate)**:
- ✅ PerformanceAnalyzer (metrics computation)
- ✅ Step comparison (step N vs N-1)
- ✅ 기본 통계 (Sharpe, volatility, drawdown, turnover)

**Phase 2 (Near-term)**:
- [ ] Visualizer (heatmaps, PnL curves)
- [ ] Color scheme support (Korean/US)
- [ ] Attribution waterfall charts

**Phase 3 (Future)**:
- [ ] Factor analysis (exposure, IC)
- [ ] Sensitivity analysis (diff feature)
- [ ] Regime analysis
- [ ] Monte Carlo simulation

