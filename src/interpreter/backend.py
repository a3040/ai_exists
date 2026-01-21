import os
import torch
import requests
import json
from abc import ABC, abstractmethod
from typing import List, Optional

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        pass

class LocalTransformersBackend(LLMBackend):
    def __init__(self, model_path: str):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "❌ [LocalTransformersBackend] 'transformers' or 'peft' library not found. "
                "Please install them via 'pip install transformers peft accelerate' to use local models."
            )
            
        print(f"📥 [Backend] Loading local transformers model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.device = self.model.device

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text.replace(prompt, "").strip()

class OllamaBackend(LLMBackend):
    def __init__(self, model_name: str = "qwen2.5:1.5b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.url = f"{host}/api/generate"
        print(f"🌐 [Backend] Ollama model: {model_name} (Matching aipcmonitoring spec)")

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0, # Focused diagnostic output
                "num_predict": max_new_tokens,
                "stop": ["Sequence:", "Input:", "\n\n"]
            }
        }
        try:
            res = requests.post(self.url, json=payload, timeout=15)
            res.raise_for_status()
            content = res.json().get('response', '').strip()
            
            # Robust parsing for small models (like 1.5b) that might add prefixes
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            if not lines:
                return "상태 데이터 분석 중..."
            
            for line in lines:
                for prefix in ["Report:", "진단:", "결과:", "보고:", "해석:"]:
                    if prefix in line:
                        return line.split(prefix)[-1].strip()
            
            return lines[-1] # Return last meaningful line
        except Exception as e:
            print(f"❌ [OllamaBackend] Error: {e}")
            return f"Ollama Connection Error: {str(e)}"

def get_backend() -> LLMBackend:
    # Factory to select backend based on environment
    # SWITCHED TO TRANSFORMERS (Local Fine-tuned Model)
    mode = os.getenv("LLM_MODE", "transformers").lower()
    if mode == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        return OllamaBackend(model_name=model_name)
    else:
        # Default path for the fine-tuned model (Local or relative)
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/finetuned_qwen"))
        model_path = os.getenv("MODEL_PATH", default_path)
        return LocalTransformersBackend(model_path=model_path)
