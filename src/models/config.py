from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    context_size: int
    vocab_size: int
    
    d_embed: int
    n_layer: int
    n_head: int
    
    dropout: float = 0.1
    
    kernel_function: str = 'softmax'
    use_ff: bool = True
    use_ppe: bool = False
    use_nto: bool = False # Only predict the N+1 token
    
    def __post_init__(self):
        assert self.kernel_function in ['softmax', 'linear', 'rbf', 'laplacian']
        self.d_ff = self.d_embed * 4