import torch

class PhyTokenMapper:
    def __init__(self):
        self.vocab = {
            "<PHY_IDLE>",
            "<PHY_LOAD_RISING>",
            "<PHY_LOAD_SATURATED>",
            "<PHY_STATE_SHIFT>",
            "<PHY_ANOMALY>"
        }
        self.high_load_threshold = 0.8

    def map(self, abstract_state: torch.Tensor, entropy: float, state_flag: str) -> str:
        """
        L3-4.2 map() logic
        """
        # abstract_state.load check (conceptual, using norm as proxy if not defined)
        # In a real scenario, abstract_state would have a specific dimension for 'load'
        load_value = torch.norm(abstract_state).item()
        
        if state_flag == "STABLE" and load_value > self.high_load_threshold:
            return "<PHY_LOAD_SATURATED>"
            
        if state_flag == "TRANSITION":
            return "<PHY_STATE_SHIFT>"
        
        if load_value > 0.1:
            return "<PHY_LOAD_RISING>"
            
        return "<PHY_IDLE>"
