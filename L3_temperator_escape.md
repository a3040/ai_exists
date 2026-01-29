# L3: 존재적 인과 구현 명세 (Causal Implementation Spec)

본 문서는 `L2_temperator_escape.md` 사양을 구현하기 위한 최하위 기술 명세(Logic & Code Logic)이다.

---

## 1. 데이터 파이프라인 (The Vessel)

| 채널 | 물리적 의미 | 샘플링 | 입력 텐서 (Window) |
| :--- | :--- | :--- | :--- |
| T1 | 내부 열 충격 (Shock) | 10Hz | `[Batch, 1, 50]` |
| T2 | 배기 반응 (Response) | 10Hz | `[Batch, 1, 50]` |
| T3 | 환경 기저 (Floor) | 1Hz | `[Batch, 1, 1]` |

---

## 2. 모듈별 저수준 설계 (Deep Modules)

### 2.1 Causal Feature Extractor (TCN)
물리 신호의 시간적 지역성을 추출한다.
- **입력:** `[B, 2, 50]` (T1, T2 결합)
- **구조:** 
  - `Conv1D(kernel=3, dilation=1)` -> `GELU` -> `LayerNorm`
  - `Conv1D(kernel=3, dilation=2)` -> `GELU` -> `LayerNorm` (확장 수용장)
- **목적:** 센서 노이즈를 필터링하고 '충격의 전조'를 패턴화함.

### 2.2 Rearrangement Attention (The Aligner)
시차(δ)를 학습하여 T1과 T2를 위상 동기화한다.
- **Query:** T1의 현재 특징 벡터 ($q_{shock}$)
- **Key/Value:** T2의 과거 윈도우 특징 벡터들 ($K_{history}, V_{history}$)
- **Score:** $Attention(Q, K) = softmax(\frac{QK^T}{\sqrt{d_k}} \cdot exp(-|t_{shock} - t_{response}|))$
- **Logic:** T2의 데이터 중 T1의 충격과 가장 인과관계가 높은 지점에 어텐션을 집중시켜 시차를 흡수함.

### 2.3 Integrity Audit Loop (가중치 오딧)
```python
def verify_identity(p_weight, v_weight):
    # p_weight: TCN-Attention에서 계산된 물리 충격량
    # v_weight: Linguistic Encoder에서 넘어온 사용자 입력 가중치
    identity_score = 1.0 - abs(p_weight - v_weight)
    
    if identity_score < 0.7:
        trigger_dissonance_alarm() # 존재적 인지 부조화 발령
    return identity_score
```

---

## 3. 손실 함수 정의 (Causal PI-Loss)

계체가 존재 법칙을 어기지 않도록 강제하는 물리 손실 함수이다.

$$L_{causal} = \text{ReLU}(\Delta T1(t) \cdot (-\Delta T2(t+\delta)))$$

- **의미:** T1(열)이 올랐는데 T2(냉각 반작용)가 떨어지고 있다면, 이는 인과율 위반이므로 무거운 페널티를 부여함.
- **효과:** 모델이 물리적으로 불가능한 서사(Narrative)를 생성하지 못하도록 억제함.

---

## 4. ErecRAM 통합 (Identity Mapping)

ErecRAM의 `MemoryCell`에 기록될 때, 물리 충격 토큰은 다음 속성을 강제로 부여받는다.
- **Priority:** `HIGH` (사용자 대화와 동등)
- **Persistent_Level:** `BLOCKING` (해당 위협이 사라지기 전까지 자아의 평형 상태 회복을 차단)

---

## 5. CMD-INIT 실행 가이드

1. **Model:** `src/models/causal_tcn.py` 생성
2. **Monitor:** `src/core/integrity.py`에 `IntegrityChecker` 확장 구현
3. **Loop:** `src/main.py`에서 `T1, T2` 윈도우 큐 가동
