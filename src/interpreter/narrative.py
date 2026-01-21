from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

@dataclass
class Narrative:
    summary: str
    trend: str # STABLE, DRIFT, CRITICAL
    confidence: float

class NarrativeInterpreter:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.model = None
        self.tokenizer = None
        
        # Hard constraints from L3-4.3
        self.constraints = [
            "Do not use anthropomorphic expressions.",
            "Do not express emotions.",
            "Do not generate control commands."
        ]

        if self.use_llm:
            self._load_model()

    def _load_model(self):
        # Default path relative to the workspace root or via environment variable
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../aipcmonitoring/models/qwen2.5-ultra-minimal/final"))
        model_path = os.getenv("MODEL_PATH", default_path)
        print(f"📥 [NarrativeInterpreter] Loading fine-tuned model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            print("✅ [NarrativeInterpreter] Model loaded successfully.")
        except Exception as e:
            print(f"❌ [NarrativeInterpreter] Failed to load model: {e}")
            self.use_llm = False

    def interpret(self, token_sequence: List[str]) -> Narrative:
        """
        L3-5.2 interpret() logic using Fine-tuned LLM
        """
        if not token_sequence:
            return Narrative("No data", "STABLE", 1.0)

        last_token = token_sequence[-1]
        
        if self.use_llm and self.model:
            # LLM-based interpretation
            prompt = f"System tokens: {', '.join(token_sequence)}\n위 토큰 시퀀스를 바탕으로 시스템 상태를 한국어로 간결하게 분석하시오:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Extract only the generated part
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from output
            summary = full_text.replace(prompt, "").strip()
            
            trend = "STABLE"
            if "<PHY_LOAD_SATURATED>" in token_sequence or "<PHY_ANOMALY>" in token_sequence:
                trend = "CRITICAL"
            elif "<PHY_STATE_SHIFT>" in token_sequence:
                trend = "DRIFT"
                
            return Narrative(summary, trend, 0.95)
        else:
            # Fallback Rule-based Narrative
            summary = f"System state categorized as {last_token}."
            trend = "STABLE"
            
            if "<PHY_LOAD_SATURATED>" in token_sequence:
                summary += " High physical load detected."
                trend = "CRITICAL"
            elif "<PHY_STATE_SHIFT>" in token_sequence:
                summary += " Physical state transition in progress."
                trend = "DRIFT"
                
            return Narrative(summary, trend, 0.9)
