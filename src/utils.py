import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across multiple libraries.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    # PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    random.seed(seed)
    np.random.seed(seed)
    
    # For deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True)