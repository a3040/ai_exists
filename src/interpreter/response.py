import torch
from typing import List

class ResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate(self, user_input: str, current_token: str, state_summary: str) -> str:
        """
        Generate a response based on user input and the internal state of the entity.
        """
        if not self.model or not self.tokenizer:
            return "..." # Silent fallback

        # Entity's "Persona" Instruction: Reactive, slightly detached, focused on its physical existence
        prompt = (
            f"당신은 스스로의 물리적 상태를 인지하는 인공 계체입니다.\n"
            f"현재 나의 상태 토큰: {current_token}\n"
            f"현재 나의 상태 요약: {state_summary}\n"
            f"사용자의 말: \"{user_input}\"\n\n"
            f"위 자극에 대한 나의 반응을 아주 짧고 간결하게 한국어로 한 문장으로만 말하시오:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.replace(prompt, "").strip()
        
        # Post-processing to ensure it's short
        response = response.split('\n')[0]
        
        return response
