# 2\. 아키텍처 (Architecture)

PRD의 요구사항(F1, F2, F3)을 구현하기 위해 **퍼사드(Facade)**, **컴포짓(Composite)**, **비지터(Visitor)** 디자인 패턴을 기반으로 시스템을 설계합니다.

## 2.1. 핵심 컴포넌트

* **A. `AlphaLab` (`rc` 객체): 퍼사드 (Facade) 🏛️**

  * **역할:** `rc` 객체는 "최상위 컨트롤러"이자 사용자를 위한 \*\*단일 통합 인터페이스(Facade)\*\*입니다. "홈시어터 퍼사드"가 `DvdPlayer`, `Projector`, `Amplifier` 등 복잡한 서브시스템을 지휘하듯, `rc`는 다음과 같은 내부 컴포넌트들을 지휘하고 조율합니다.
  * **소유 컴포넌트:**
        1. **`rc.db` (State):** `xarray.Dataset` 인스턴스. 모든 `(T, N)` 데이터(원본, 버킷, 시그널)가 `DataArray`로 저장되는 핵심 "데이터 저장소"입니다.
        2. **`rc.rules` (Registry):** `add_axis`로 정의된 "룰"(`Expression` 객체)을 저장하는 `dict`입니다.
        3. **`rc._evaluator` (Executor):** `EvaluateVisitor`의 인스턴스. `Expression` "레시피"를 실행하는 "실행자"입니다.
        4. (기타) `ConfigLoader`, `PnLTracer` 등

* **B. `Expression` 트리: 컴포짓 (Composite) 📜**

  * **역할:** "계산법" 또는 "레시피"입니다. **컴포짓 패턴**을 따르는 데이터 구조입니다.
  * **구조:**
    * `Expression`은 모든 연산 노드의 추상 인터페이스입니다.
    * **Leaf (리프):** `Field('close')`와 같이 자식이 없는 노드입니다.
    * **Composite (복합):** `ts_mean(Field('close'), 10)`와 같이 다른 `Expression` 노드를 자식으로 갖는 트리(Tree) 구조입니다.
  * **특징:** `Expression` 객체는 실제 데이터(`(T, N)` 배열)를 전혀 가지지 않고, "계산 룰"에 대한 정의만 가집니다.

* **C. `Visitor` 패턴: 실행 및 추적 (Visitor) 👨‍🍳**

  * **역할:** `Expression` 트리(레시피)를 "방문(visit)"하며 실제 작업을 수행하는 "실행자"입니다. **`Expression` 객체와는 완전히 분리된 별개의 클래스**입니다.
  * **`EvaluateVisitor`:** `rc` 객체(`rc._evaluator`)가 소유하며, `Expression` 트리를 순회하며 `rc.db`의 데이터를 참조하여 실제 `xarray.DataArray`를 계산합니다.

## 2.2. 기능별 아키텍처 구현

* **F1 (데이터 검색):**

    1. `rc.add_data('close', Field('price_close'))` 호출 시, `rc`는 `Field('price_close')`를 `rc.rules`에 등록합니다.
    2. 이후 `EvaluateVisitor`가 `Field('price_close')` 노드를 방문하면, `ConfigLoader`를 호출하여 `data_config.yaml`을 읽고 SQL을 실행합니다.
    3. 결과 `(T, N)` `DataArray`를 `rc.db['close']`에 저장(캐시)합니다.

* **F2 (셀렉터 인터페이스):**

    1. `rc.add_axis('size', cs_quantile(rc.data.mcap, ...))` 호출 시, `rc`는 이 `cs_quantile` `Expression` 객체를 `rc.rules['size']`에 등록합니다.
    2. 사용자가 `mask = rc.axis.size['small']`을 호출합니다.
    3. `rc`는 `rc.rules['size']`에서 `Expression`을 꺼내 `Equals(..., 'small')`이라는 새 `Expression`을 만듭니다.
    4. `rc._evaluator` (Visitor)에게 이 `Expression`을 평가(evaluate)하도록 지시합니다.
    5. `EvaluateVisitor`가 `(T, N)` 불리언 마스크를 반환합니다.
    6. 사용자가 `rc[mask] = 1.0`을 호출하면, `rc`는 `rc.db['my_alpha']` 캔버스에 `xr.where`를 사용하여 값을 할당(overwrite)합니다.

* **F3 (심층 추적성):**

    1. `rc._evaluator` (Visitor)는 **"Stateful(상태 저장)"** 객체입니다.
    2. `EvaluateVisitor`는 내부에 `cache` (e.g., `dict`)를 소유합니다.
    3. `rc.add_data_var('alpha1', ...)`가 호출되면, `Visitor`는 `alpha1`의 `Expression` 트리를 순회하면서 **각 노드가 반환하는 중간 결과 `(T, N) DataArray`를 `self.cache`에 모두 저장**합니다. (e.g., `cache['returns']`, `cache['ts_mean(...)']` 등)
    4. 사용자가 `rc.trace_pnl('alpha1')`를 호출하면, `rc`는 `rc._evaluator.cache`에 저장된 `(T, N)` 배열들을 순서대로 꺼내 `PnLTracer`에게 전달합니다.
    5. `PnLTracer`는 **재계산 없이(no re-computation)** 이 배열들로 PnL을 계산하여 사용자에게 리포트합니다.
