import psutil
import torch
import time

class PhysicalBody:
    def __init__(self):
        print("🔧 [PhysicalBody] Initializing real-world sensing (psutil)...")
        # Ensure initial measurement to avoid 0.0 reading on first call
        psutil.cpu_percent(interval=None)
        
    def sense(self) -> torch.Tensor:
        """
        Extract real system metrics and return as a tensor.
        Dimensions: [CPU_Usage, Memory_Usage, Disk_Usage, Swap_Usage, IO_Wait (approx)]
        """
        cpu = psutil.cpu_percent(interval=None) / 100.0
        memory = psutil.virtual_memory().percent / 100.0
        disk = psutil.disk_usage('/').percent / 100.0
        swap = psutil.swap_memory().percent / 100.0
        
        # IO wait is often tricky, using a simpler metric here
        # or just 4 metrics for now to match main.py's INPUT_DIM=5 (padding one)
        # Let's use CPU, RAM, Disk, Swap, and 0.0 as filler or a random jitter
        
        state = torch.tensor([cpu, memory, disk, swap, 0.0], dtype=torch.float32)
        return state.unsqueeze(0) # [1, 5]
