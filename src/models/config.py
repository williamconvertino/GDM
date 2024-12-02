from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    
    model_type: str # 'GPT' or 'GDM'
    
    context_size: int
    vocab_size: int
    
    d_embed: int
    n_layer: int
    n_head: int
    
    dropout: float = 0.1
    
    attn_kernel_fn: str = 'softmax'
    use_ff: bool = True
    use_ppe: bool = False
    use_nto: bool = False # Only predict the N+1 token
    
    def __post_init__(self):
        assert self.model_type in ['GPT', 'GDM']
        assert self.attn_kernel_fn in ['softmax', 'linear', 'rbf', 'laplacian']
        self.d_ff = self.d_embed * 4