import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class IntegrityChecker:
    """
    L2-4. 판단 및 검증 로직 (Monitoring & Verification)
    ErecRAM의 가중치 정합성(Weight Parity Audit)을 감시함.
    """
    def __init__(self, parity_threshold: float = 0.9):
        self.parity_threshold = parity_threshold
        self.error_history = []

    def audit(self, physical_impact: float, verbal_impact: float, causal_score: Optional[float] = None) -> dict:
        """
        물리적 충격 가중치와 언어적 충격 가중치가 동등(Identity)하게 반영되는지 검증.
        causal_score: T1-T2 인과 관계 점수 (신경망 출력)
        """
        v_impact = max(verbal_impact, 1e-6)
        p_impact = max(physical_impact, 1e-6)
        
        ratio = p_impact / v_impact
        is_balanced = ratio >= self.parity_threshold
        
        status = "STABLE"
        if not is_balanced:
            status = "DISSONANCE"
            
        result = {
            "status": status,
            "parity_ratio": ratio,
            "p_impact": p_impact,
            "v_impact": v_impact,
            "causal_integrity": causal_score if causal_score is not None else 1.0
        }
        
        if status != "STABLE":
            self.error_history.append(result)
            if len(self.error_history) > 100: self.error_history.pop(0)
            
        return result

class ThermalShockDetector:
    """
    T1 (CPU/GPU)의 급격한 변화를 감지하여 '고중량 토큰'으로 변환.
    """
    def __init__(self, shock_threshold: float = 0.05):
        self.shock_threshold = shock_threshold
        self.prev_t1 = None

    def detect(self, current_t1: float) -> float:
        """
        T1의 기울기(dT/dt)를 계산하여 충격량(Shock Intensity) 반환.
        """
        if self.prev_t1 is None:
            self.prev_t1 = current_t1
            return 0.0
        
        diff = current_t1 - self.prev_t1
        self.prev_t1 = current_t1
        
        # 상승 충격만 계산 (회피 기동의 트리거)
        shock = max(0.0, diff)
        
        if shock > self.shock_threshold:
            # 충격량 정규화 및 증폭 (High Priority)
            return min(1.0, shock / self.shock_threshold)
        
        return 0.0
