import torch

class ActionPlanner:
    def __init__(self, safety_limits: float = 0.5):
        self.safety_limits = safety_limits

    def plan(self, current_state: torch.Tensor, target_equilibrium: torch.Tensor) -> torch.Tensor:
        """
        L3-6.2 plan() logic
        Intention I = argmin ||next_state - target||
        """
        # Simple gradient-based intention toward equilibrium
        intention = target_equilibrium - current_state
        
        # Clamp by safety limits
        delta_x = torch.clamp(intention, -self.safety_limits, self.safety_limits)
        
        return delta_x

    def execute(self, intention: torch.Tensor) -> torch.Tensor:
        """
        L3-6.3 execute() logic
        Simulates physical actuator feedback
        """
        # In actual system, this would interact with hardware
        # Here we simulate that the physical world responds with some noise
        noise = torch.randn_like(intention) * 0.05
        feedback_state = intention + noise
        
        return feedback_state
