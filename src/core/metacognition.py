import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional
from .nas_erec import NASErecRAM

class MetacognitionMonitor:
    def __init__(self, ram: NASErecRAM):
        self.ram = ram
        self.analysis_history: List[Dict] = []
        self.sleep_mode = False
        self.sleep_threshold = 0.7
        self.fatigue = 0.0

    def monitor(self) -> Dict:
        """
        ErecRAM의 내부 상태를 냉철하게 분석한다.
        """
        if not self.ram.memory_bank:
            return {"status": "empty"}

        # 1. Consistency (현재 상태와 기억들의 유사도)
        # Ensure device consistency for GPU support
        device = self.ram.current_state.device
        states = torch.stack([cell.state for cell in self.ram.memory_bank]).to(device)
        weights = torch.tensor([cell.weight for cell in self.ram.memory_bank]).to(device)
        
        # Normalize weights
        norm_weights = F.softmax(weights, dim=0)
        
        # Cosine similarity between current_state and memory cells
        cos_sim = F.cosine_similarity(self.ram.current_state.unsqueeze(0), states, dim=1)
        mean_similarity = torch.sum(cos_sim * norm_weights).item()

        # 2. Attention Focus (메모리 가중치의 엔트로피 -> 협소한지, 분산되어 있는지)
        # Low entropy = focused on a few memories
        # High entropy = confused/all memories equally weighted
        entropy = -torch.sum(norm_weights * torch.log(norm_weights + 1e-9)).item()
        max_entropy = math.log(len(self.ram.memory_bank))
        relative_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # 3. State Energy (Magnitude of the current state)
        energy = torch.norm(self.ram.current_state).item()

        # 4. Fatigue (자극 누적도)
        # 신규 입력이 계속 들어오면 fatigue 상승, sleep 시 하강
        # 4096차원에서는 노이즈가 많으므로 증가치를 대폭 낮춤 (0.05 -> 0.005)
        self.fatigue += (1.0 - mean_similarity) * 0.005
        self.fatigue = max(0.0, min(1.0, self.fatigue))

        analysis = {
            "mean_similarity": mean_similarity,
            "relative_entropy": relative_entropy,
            "energy": energy,
            "fatigue": self.fatigue,
            "memory_count": len(self.ram.memory_bank)
        }
        
        self.analysis_history.append(analysis)
        if len(self.analysis_history) > 100:
            self.analysis_history.pop(0)
            
        return analysis

    def evaluate_consciousness(self, analysis: Dict) -> str:
        """
        분석 결과를 바탕으로 계체의 의식 상태를 판정한다.
        """
        if analysis.get("status") == "empty":
            return "INITIALIZING"

        sim = analysis["mean_similarity"]
        ent = analysis["relative_entropy"]
        fatigue = analysis["fatigue"]

        if fatigue > 0.8:
            return "OVERLOADED"
        if sim > 0.9 and ent < 0.3:
            return "HYPER_FOCUSED"
        if sim < 0.4:
            return "DISCONNECTED"
        if ent > 0.8:
            return "CONFUSED"
        
        return "STABLE"

    def should_sleep(self) -> bool:
        """
        계체가 휴식(수면)이 필요한지 결정한다.
        """
        if not self.analysis_history:
            return False
        
        recent = self.analysis_history[-1]
        # 수면 임계치 상향 (0.7 -> 0.85)
        if recent["fatigue"] > 0.85:
            return True
        return False

    def consolidate(self):
        """
        수면(Sleep) 모드에서 수행되는 정보 통합 및 피로 해소 로직.
        """
        if not self.ram.memory_bank:
            return

        # 1. Low-weight memory pruning (망각: 중요도가 낮은 기억 제거)
        initial_count = len(self.ram.memory_bank)
        self.ram.memory_bank = [cell for cell in self.ram.memory_bank if cell.weight > 0.05]
        pruned_count = initial_count - len(self.ram.memory_bank)

        # Safety Check: 만약 모든 기억이 제거되었다면 중단
        if not self.ram.memory_bank:
            self.fatigue *= 0.5 # 강제 휴식 효과
            return {"pruned": pruned_count, "current_fatigue": self.fatigue, "is_sleeping": self.sleep_mode}

        # 2. State Stabilization (현재 상태를 메모리의 가중 평균 방향으로 정렬)
        device = self.ram.current_state.device
        states = torch.stack([cell.state for cell in self.ram.memory_bank]).to(device)
        weights = torch.tensor([cell.weight for cell in self.ram.memory_bank]).to(device)
        norm_weights = F.softmax(weights, dim=0).unsqueeze(1)
        
        summary_state = torch.sum(states * norm_weights, dim=0)
        
        # 현재 상태를 내부 요약 상태로 부드럽게 전이 (자아의 내적 정합성 강화)
        self.ram.current_state.data = (0.8 * self.ram.current_state.data) + (0.2 * summary_state)

        # 3. Fatigue recovery (피로도 회복)
        self.fatigue *= 0.85 
        
        if self.fatigue < 0.2:
            self.sleep_mode = False
            
        return {
            "pruned": pruned_count,
            "current_fatigue": self.fatigue,
            "is_sleeping": self.sleep_mode
        }

    def get_summary(self, analysis: Dict) -> str:
        status = self.evaluate_consciousness(analysis)
        sleep_str = " [SLEEPING]" if self.sleep_mode else ""
        summary = f"[Metacog]{sleep_str} Stat: {status} | Sim: {analysis.get('mean_similarity', 0):.3f} | Ent: {analysis.get('relative_entropy', 0):.3f} | Fatigue: {analysis.get('fatigue', 0):.3f}"
        return summary

    # --- NAS Controller Logic ---

    def search_architecture(self, analysis: Dict) -> Optional[Dict[str, str]]:
        if not hasattr(self.ram, 'dna'):
            return None 

        # If entropy is sustained high (~0.9) over 20 steps, trigger search
        recent_entropy = [a.get('relative_entropy', 0) for a in self.analysis_history[-20:]]
        if len(recent_entropy) >= 20 and sum(recent_entropy)/20 > 0.85:
            return self._propose_mutation()
        
        return None

    def _propose_mutation(self) -> Dict[str, str]:
        import random
        from .modules import SEARCH_SPACE
        
        current_dna = self.ram.dna.copy()
        gene_to_mutate = random.choice(list(SEARCH_SPACE.keys()))
        new_value = random.choice(list(SEARCH_SPACE[gene_to_mutate].keys()))
        
        current_dna[gene_to_mutate] = new_value
        return current_dna
