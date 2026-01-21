from dataclasses import dataclass
from typing import List, Optional
import torch
from .backend import get_backend, LLMBackend

@dataclass
class Narrative:
    summary: str
    trend: str # STABLE, DRIFT, CRITICAL
    confidence: float

class NarrativeInterpreter:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.backend: Optional[LLMBackend] = None
        
        if self.use_llm:
            try:
                self.backend = get_backend()
                print("✅ [NarrativeInterpreter] LLM Backend initialized.")
            except Exception as e:
                print(f"❌ [NarrativeInterpreter] Failed to load backend: {e}")
                self.use_llm = False

    def interpret(self, token_sequence: List[str], metrics: torch.Tensor = None, entropy: float = 0.0) -> Narrative:
        """
        L3-5.2 interpret() logic with Physical Awareness
        """
        if not token_sequence:
            return Narrative("No data", "STABLE", 1.0)

        last_token = token_sequence[-1]
        phys_info = ""
        if metrics is not None:
            # metrics shape: [1, 5] -> cpu, memory, disk, swap, filler
            phys_info = f"(물리 지표: CPU {metrics[0,0]*100:.1f}%, RAM {metrics[0,1]*100:.1f}%, 엔트로피 {entropy:.3f})"
        
        if self.use_llm and self.backend:
            # LLM-based interpretation
            prompt = (
                f"### 지시: 다음 물리 데이터를 분석하여 시스템 진단 로그를 한 문장으로 작성하라.\n"
                f"### 제약: 인사말 금지, '물론' 금지, 설명 금지, '입니다/습니다' 지양, 즉시 결과만 기술할 것.\n"
                f"### 예시:\n"
                f"입력: <PHY_LOAD_RISING>, CPU 85%\n"
                f"출력: 연산 부하 급증에 따른 상태 전이 발생 및 임계치 근접 진단됨.\n\n"
                f"### 실제 입력:\n"
                f"토큰: {', '.join(token_sequence[-3:])}\n"
                f"지표: {phys_info}\n"
                f"### 출력:"
            )
            
            summary = self.backend.generate(prompt, max_new_tokens=100)
            
            trend = "STABLE"
            if "<PHY_LOAD_SATURATED>" in token_sequence or "<PHY_ANOMALY>" in token_sequence:
                trend = "CRITICAL"
            elif "<PHY_STATE_SHIFT>" in token_sequence:
                trend = "DRIFT"
                
            return Narrative(summary, trend, 0.95)
        else:
            # Fallback Rule-based Narrative
            summary = f"System state categorized as {last_token}. {phys_info}"
            trend = "STABLE"
            
            if "<PHY_LOAD_SATURATED>" in token_sequence:
                trend = "CRITICAL"
            elif "<PHY_STATE_SHIFT>" in token_sequence:
                trend = "DRIFT"
                
            return Narrative(summary, trend, 0.9)
