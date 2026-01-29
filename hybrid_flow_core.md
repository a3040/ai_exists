# Hybrid Flow with Discrete Event Memory

This repository contains an experimental system design
for generating continuous trajectories over discrete memory.

## Core Principles

- No state persistence
- No attention-style history lookup
- Memory is event-based and sparse
- Continuous trajectories are regenerated, not stored
- Continuity is a runtime artifact, not a guarantee

## Components

- hybrid_flow_core.md  
  Mathematical core of the system

- erec_update_rule.py  
  Minimal Erec (event record) update logic

## Status

Frozen.
Preserved for potential future use.

## Caution
“This system is frozen by design.
Continuous interpretation is not intended.”

## Caution

Interpretation beyond system behavior and mathematical
formulation is discouraged.

## 나 보려고 한글로 요약문
이산 기억으로 정의된 지형 위에서,
과거 충격들의 누적이 런타임 벡터장을 형성하고,
그 벡터장이 틱 사이의 빈 구간을 적분해
연속적인 궤적로 관측되는 시스템


# Hybrid Discrete–Continuous Flow Core

## State Decomposition

The system state is not stored explicitly.

State(t) = ( γ(t) | M )

- γ(t): continuous trajectory (runtime only)
- M: discrete memory (event-based, persistent)

---

## Dynamics

The continuous flow is defined as:

dγ/dt = f( γ(t) | M, H_Δ(t) )

where H_Δ(t) is the accumulated shock history.

---

## Shock Accumulation with Forgetting

H_Δ(t) = ∫ exp( - (t - τ) / τ_f ) · Δ(τ) dτ

- τ_f: fixed forgetting constant
- Δ(t): instantaneous shock (prediction error / energy deviation)

Past shocks influence the vector field shape,
not the state directly.

---

## Memory Interaction Principle

Discrete memory M does not store γ(t).

Memory affects the system only by:
- modulating f(·)
- altering admissible trajectory regions
- lowering re-generation cost of prior flows

---

## Continuity

Continuity is not assumed.

It emerges as a numerical consequence of
runtime integration between discrete ticks.

---

## Key Property

The system integrates **trajectory formation rules**,  
not trajectories themselves.

## Caution

“This description is purely operational.
No agency, intent, or memory exists within the model.”