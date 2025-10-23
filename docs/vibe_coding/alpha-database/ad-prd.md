# Alpha-Database 제품 요구사항 문서 (Product Requirement Document)

## 1. 개요

**alpha-database**는 퀀트 리서치를 위한 **데이터 영속성 및 검색 패키지**입니다. 복잡한 데이터 수집이나 계산 로직 없이, **이미 저장된 데이터의 CRUD(Create, Read, Update, Delete) 및 쿼리**에 집중합니다.

### 역할 및 책임

1. **데이터 검색 (Data Retrieval)**: Config-driven 데이터 로딩
2. **데이터 저장 (Data Storage)**: Alpha, factor, dataset 영구 저장
3. **메타데이터 관리 (Metadata)**: 데이터 카탈로그 및 의존성 추적

### 명확히 하는 것: alpha-database가 하지 않는 것 (Out of Scope)

* ❌ **데이터 수집 (Fetching)**: 외부 API에서 데이터 다운로드는 ETL 파이프라인의 역할
* ❌ **데이터 계산**: Signal, weight, return 계산은 alpha-canvas의 역할
* ❌ **Expression 직렬화**: Expression은 alpha-canvas의 개념으로, alpha-canvas가 직렬화 책임
* ❌ **분석 및 시각화**: Performance metrics, charts는 alpha-lab의 역할

### 핵심 원칙

* **Loosely Coupled**: alpha-canvas와는 공개 API로만 통신
* **Source Agnostic**: 다양한 데이터 소스 지원 (Parquet, CSV, Excel, DB)
* **Simple Interface**: CRUD 중심의 명확한 인터페이스
* **Explicit over Implicit**: 자동화보다 명시적 제어 우선

## 2. 우선순위 및 구현 로드맵

### **P0 (MVP - Essential)**: 데이터 검색 (Data Retrieval)
1. Config-driven data loading
2. Multi-source reader facade
3. Alpha-canvas 통합 및 검증

### **P1 (Core)**: 데이터 저장 (Data Storage)
1. Dataset catalog (computed fields)
2. Alpha catalog (pre-computed alphas)
3. Factor catalog (factor returns)
4. 중앙 메타 테이블

### **P2 (Enhancement)**: 계보 추적 (Lineage Tracking)
1. 명시적 의존성 그래프 관리
2. Impact analysis tools

### **P3 (Future)**: 고급 기능
1. PostgreSQL backend
2. ClickHouse backend
3. Data quality checks
4. Plugin ecosystem

---

## 3. P0: Config-Driven Data Loading (MVP)

### 요구사항

**R1**: YAML 기반 데이터 소스 정의  
**R2**: Parameterized SQL 쿼리 지원 (날짜 범위 런타임 주입)  
**R3**: Long → Wide 자동 변환  
**R4**: xarray.DataArray 형식으로 반환  
**R5**: Backward compatibility (기존 alpha-canvas 코드 변경 최소화)

### 사용 시나리오

**시나리오 1: 독립 실행**
```
사용자가 data.yaml 설정 파일을 준비
→ DataSource 초기화
→ 필드 이름과 날짜 범위로 데이터 요청
→ (T, N) 형태의 DataArray 반환
```

**시나리오 2: Alpha-canvas 통합**
```
사용자가 DataSource 인스턴스 생성
→ AlphaCanvas 초기화 시 DataSource 주입 (Dependency Injection)
→ Field 표현식 사용 시 자동으로 DataSource 호출
→ AlphaCanvas가 데이터를 내부 Dataset에 캐싱
```

---

## 4. P0: Multi-Source Reader Architecture

### 요구사항

**R1**: 다양한 데이터 소스 통합 인터페이스  
**R2**: Config 기반 자동 Reader 선택  
**R3**: **Plugin Architecture**: 확장 가능한 Reader 시스템  
**R4**: 일관된 출력 형식 (Long DataFrame)

### Core vs Plugin Architecture

**Core Readers** (alpha-database 패키지에 포함):
- **ParquetReader**: DuckDB를 사용한 Parquet 파일 쿼리
- **CSVReader**: CSV 파일 로딩
- **ExcelReader**: 일반 Excel 파일 로딩

**Plugin Readers** (별도 설치 가능):
- **alpha-database-fnguide**: FnGuide 특수 포맷 Excel reader
- **alpha-database-bloomberg**: Bloomberg 데이터 reader
- **User-defined readers**: 사용자가 직접 구현한 커스텀 reader

### 아키텍처 철학

**Core**: 필수적이고 범용적인 기능만 포함
- 모든 사용자에게 필요
- 외부 의존성 최소화
- 높은 안정성과 backward compatibility

**Plugin**: 특수하거나 도메인 특화 기능
- 선택적 설치
- 독립적 버전 관리
- 빠른 업데이트 가능

### Plugin 등록 워크플로우

```
사용자가 플러그인 설치 (pip install alpha-database-fnguide)
→ Custom reader를 DataSource에 등록
→ Config 파일에서 등록된 reader 타입 사용
→ DataSource가 적절한 reader로 라우팅
```

---

## 5. P0: Alpha-Canvas 통합

### 요구사항

**R1**: Dependency Injection 패턴 지원  
**R2**: Backward compatibility 유지  
**R3**: 점진적 마이그레이션 가능

### 통합 시나리오

```
Phase 1 (Backward Compatible):
  사용자가 기존 방식대로 AlphaCanvas 초기화
  → 내부 data loader 사용 (alpha-canvas 내장)
  OR
  사용자가 DataSource를 명시적으로 주입
  → alpha-database 사용

Phase 2 (Migration):
  모든 테스트를 alpha-database로 전환
  → 안정성 검증

Phase 3 (Cleanup):
  alpha-canvas 내장 loader 제거
  → alpha-database 필수 의존성으로 변경
```

---

## 6. P1: Dataset Catalog

### 요구사항

**R1**: Schema evolution (컬럼 동적 추가)  
**R2**: **명시적 의존성** (사용자가 제공)  
**R3**: Upsert 로직 (날짜 중복 시 덮어쓰기)  
**R4**: 메타데이터 자동 관리

### 사용 시나리오

**시나리오 1: 파생 지표 생성**
```
리서처가 alpha-canvas에서 PBR 계산
→ 결과 DataArray와 의존성 정보(adj_close, book_value) 준비
→ DataWriter로 'fundamental' dataset에 'pbr' 필드 저장
→ 메타 테이블 자동 업데이트
→ 이후 다른 연구에서 'pbr' 필드 재사용 가능
```

**시나리오 2: Schema Evolution**
```
기존 'fundamental' dataset에 'pbr' 필드 존재
→ 리서처가 EV/EBITDA 계산
→ 동일 dataset에 'ev_ebitda' 필드 추가
→ Dataset schema 자동 확장 [date, asset, pbr] → [date, asset, pbr, ev_ebitda]
→ 기존 데이터 보존, 새 컬럼만 추가
```

**시나리오 3: 재사용**
```
리서처가 여러 알파 전략 개발
→ 모두 'fundamental.pbr' 필드 사용
→ 한 번 계산, 여러 번 재사용
→ 계산 시간 절약 및 일관성 보장
```

---

## 7. P1: Alpha Catalog

### 요구사항

**R1**: 알파 결과 데이터 저장 (signal, weights, returns)  
**R2**: **Expression 저장** (alpha-canvas가 직렬화, alpha-database는 저장만)  
**R3**: **명시적 의존성** (사용자 제공)  
**R4**: 자동 버전 관리  
**R5**: 메타데이터 저장 (description, tags 등)

### 사용 시나리오

**시나리오 1: 알파 저장**
```
리서처가 alpha-canvas에서 momentum 전략 개발
→ Expression을 평가하여 signal, weights, returns 획득
→ alpha-canvas의 직렬화 기능으로 Expression을 dict 형태로 변환
→ alpha-canvas의 의존성 추출 기능으로 필요 필드 목록 획득
→ 모든 데이터와 메타데이터를 DataWriter에 전달
→ 자동 버전 관리 (momentum_v1, momentum_v2, ...)
→ 메타 테이블 자동 업데이트
```

**시나리오 2: 알파 재현**
```
다른 리서처가 저장된 알파 조회
→ DataReader로 'momentum_v1' 메타데이터 로드
→ Expression dict 추출
→ alpha-canvas의 역직렬화 기능으로 Expression 재구성
→ 동일한 Expression으로 재평가
→ 원본과 동일한 결과 획득 (완전한 재현성)
```

**시나리오 3: 버전 비교**
```
리서처가 momentum 전략 개선
→ 새 버전 저장 (자동으로 momentum_v2 생성)
→ alpha-lab으로 v1과 v2 성과 비교
→ 더 나은 버전 선택
→ 개선 과정 이력 보존
```

---

## 8. P1: Factor Catalog

### 요구사항

**R1**: 팩터 수익률 time series 저장  
**R2**: Expression 정보 저장 (메타데이터)  
**R3**: 명시적 의존성  
**R4**: 선택적 버전 관리

### 사용 시나리오

**시나리오: Fama-French SMB 팩터 저장**
```
리서처가 size factor 계산 로직 구현
→ alpha-canvas로 포트폴리오 구성 및 수익률 계산
→ 팩터 수익률 (T,) time series 획득
→ Expression 및 의존성 정보 준비
→ DataWriter로 'fama_french_smb' 저장
→ 메타데이터에 construction methodology 기록
→ 이후 팩터 회귀 분석이나 리스크 모델에 재사용
```

---

## 9. P1: Meta Table (Central Metadata Registry)

### 요구사항

**R1**: 모든 데이터 엔티티 통합 관리  
**R2**: 단순한 Parquet 테이블 (ORM 불필요)  
**R3**: 의존성 정보 저장 (명시적)  
**R4**: Tag 기반 검색

### 사용 시나리오

**시나리오 1: 카탈로그 탐색**
```
리서처가 사용 가능한 모든 알파 조회
→ Meta Table에서 type='alpha' 필터링
→ 알파 목록, 설명, 태그 확인
→ 관심 있는 알파 선택
```

**시나리오 2: 의존성 추적**
```
리서처가 'adj_close' 필드 업데이트 계획
→ Meta Table에서 'adj_close'에 의존하는 모든 엔티티 조회
→ 영향받는 알파/팩터 목록 확인 (pbr, momentum_v1, ...)
→ 재계산 필요 여부 판단
```

**시나리오 3: Tag 검색**
```
리서처가 momentum 관련 전략 검색
→ Meta Table에서 tags에 'momentum' 포함된 엔티티 검색
→ 관련 알파들 발견
→ 유사 전략 비교 연구
```

---

## 10. P2: Lineage Tracking

### 요구사항

**R1**: 명시적 의존성 그래프 관리  
**R2**: Impact analysis  
**R3**: Lineage visualization (future)

### 의존성 구조 예시

```
Raw Fields (데이터 소스):
  - adj_close
  - book_value
  - returns

Derived Fields (계산된 필드):
  - pbr (depends on: adj_close, book_value)
  - ev_ebitda (depends on: ev, ebitda)

Alphas (전략):
  - momentum_v1 (depends on: returns)
  - value_v1 (depends on: pbr, ev_ebitda)

Factors (팩터):
  - fama_french_smb (depends on: adj_close, book_value)
```

### 사용 시나리오

**시나리오 1: Forward Impact**
```
'returns' 필드 데이터 업데이트 예정
→ LineageTracker로 영향받는 엔티티 조회
→ 결과: [momentum_v1, momentum_v2, ...]
→ 재계산 우선순위 결정
```

**시나리오 2: Backward Trace**
```
'momentum_v1' 알파 검증 중
→ LineageTracker로 의존 필드 조회
→ 결과: ['returns']
→ 원천 데이터 품질 확인
```

---

## 11. 사용자 워크플로우 (User Workflows)

### 워크플로우 1: 일일 퀀트 리서치

```
[Morning]
리서처가 DataSource로 최신 가격 데이터 로드
→ alpha-canvas로 새로운 알파 아이디어 탐색
→ 여러 Expression 조합 실험

[Afternoon]
유망한 알파 발견
→ alpha-canvas로 백테스트 실행
→ 결과 (signal, weights, returns) 확인
→ Expression 직렬화 및 의존성 추출
→ alpha-database에 저장 (자동 버전 관리)

[Evening]
alpha-lab으로 저장된 알파 분석
→ 성과 지표 계산 (Sharpe, drawdown, ...)
→ 다른 알파들과 비교
→ 최종 후보 선정
```

### 워크플로우 2: 팀 협업

```
[Researcher A]
파생 지표 (PBR, EV/EBITDA) 계산
→ 'fundamental' dataset에 저장
→ 의존성 명시 (adj_close, book_value, ...)
→ 팀 공유 저장소에 커밋

[Researcher B]
Meta Table에서 'fundamental' dataset 발견
→ PBR 필드 로드
→ value 전략 개발에 활용
→ 계산 중복 없음, 일관성 보장

[Team Lead]
Meta Table에서 모든 알파 현황 조회
→ Tag별 분류 (momentum, value, quality, ...)
→ 포트폴리오 다각화 계획
→ 각 알파의 의존성 확인하여 리스크 관리
```

### 워크플로우 3: 프로덕션 파이프라인

```
[ETL Pipeline (외부)]
외부 데이터 소스에서 일일 데이터 수집
→ Parquet 파일로 저장
→ data.yaml 설정 업데이트

[Alpha Computation (alpha-canvas)]
DataSource로 최신 데이터 로드
→ 프로덕션 알파들 평가
→ 포트폴리오 가중치 계산
→ 결과 저장 (alpha-database)

[Risk Management (alpha-lab)]
저장된 결과 로드
→ 일일 성과 모니터링
→ 이상 징후 감지
→ 리포트 생성

[Portfolio Management]
Meta Table에서 활성 알파 조회
→ 각 알파의 최신 가중치 로드
→ 실제 거래 실행
```

---

## 12. 아키텍처 단순화 결정

### 제외된 기능과 그 이유

**1. Data Fetching (외부 API 수집)**
- **이유**: 데이터 수집은 ETL의 역할, DB의 역할 아님
- **대안**: 사용자가 별도 ETL 스크립트 작성 → Parquet 저장 → alpha-database로 로드

**2. Expression Auto-Serialization**
- **이유**: Expression은 alpha-canvas의 핵심 개념
- **대안**: alpha-canvas가 직렬화 메서드 제공, alpha-database는 결과만 저장

**3. Automatic Dependency Extraction**
- **이유**: Expression 파싱은 복잡하고 강결합 유발
- **대안**: 사용자가 명시적으로 dependencies 제공, alpha-canvas가 helper 메서드 제공

**4. Computation Cache Management**
- **이유**: 계산 캐시는 alpha-canvas의 책임
- **대안**: alpha-canvas가 캐시 관리, alpha-database는 최종 결과만 저장

**5. Built-in Specialized Readers**
- **이유**: 유지보수 부담, 모든 특수 포맷 지원 불가능
- **대안**: Core readers만 내장, 특수 readers는 plugin으로 제공

---

## 13. MVP 구현 우선순위

### Phase 1: Data Retrieval (P0) - 2-3 weeks
1. ConfigLoader 구현 (alpha-database 자체 config)
2. DataLoader 구현 (Long→Wide pivoting)
3. DataSource facade
4. Core readers (Parquet, CSV, Excel)
5. Plugin registration system
6. Alpha-canvas 통합 (dependency injection)
7. Backward compatibility validation

### Phase 2: Data Storage (P1) - 2-3 weeks
1. DataWriter 구현 (단일 인터페이스)
2. Meta Table 구현
3. LineageTracker 구현 (명시적 의존성)
4. Versioning logic
5. Schema evolution

### Phase 3: Lineage & Quality (P2) - 1-2 weeks
1. Lineage visualization tools
2. Data quality checks (optional)

---

**Total Estimated Effort**: 5-8 weeks for P0-P2 (Simplified MVP)

**Core Principle**: **alpha-database는 데이터 CRUD와 쿼리에만 집중합니다. 데이터 수집, 계산, 직렬화는 다른 컴포넌트의 책임입니다.**
