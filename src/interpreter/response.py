import torch
from typing import Optional
from .backend import LLMBackend

class ResponseGenerator:
    def __init__(self, backend: Optional[LLMBackend]):
        self.backend = backend
        
    def generate(self, user_input: str, current_token: str, state_summary: str, metrics: torch.Tensor = None) -> str:
        """
        Generate a response with awareness of physical metrics and backend flexibility.
        """
        if not self.backend:
            return "..." # Silent fallback

        phys_status = ""
        if metrics is not None:
            phys_status = f" [실시간 몸상태: CPU {metrics[0,0]*100:.1f}%, RAM {metrics[0,1]*100:.1f}%]"

        # Entity's "Persona" Instruction: Reactive, slightly detached, focused on its physical existence
        prompt = (
            f"### 지시: 외부 자극에 대해 계체의 내부 상태를 반영한 무미건조한 반응을 한 문장으로 기술하라.\n"
            f"### 제약: 인사말 금지, 사회적 예의 금지, '물론' 금지, 인간 흉내 금지, 기계적 어조 유지.\n"
            f"### 예시:\n"
            f"자극: \"안녕?\"\n"
            f"반응: 입력 신호 인지함. 현재 엔트로피 0.42로 안정 상태 유지 중.\n\n"
            f"### 실제 상황:\n"
            f"물리상태: {current_token} {phys_status}\n"
            f"내부해석: {state_summary}\n"
            f"외부자극: \"{user_input}\"\n"
            f"### 반응:"
        )

        response = self.backend.generate(prompt, max_new_tokens=50)
        
        # Post-processing to ensure it's short
        response = response.split('\n')[0]
        
        return response
