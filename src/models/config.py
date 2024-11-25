from typing import Optional
from dataclasses import dataclass

@dataclass
class GPTConfig:
    context_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    d_embed: int
    d_attn: Optional[int] = None # If None, defaults to d_embed
    d_ff: Optional[int] = None # if None defaults to 4x d_embed
    use_attn: bool = True
    use_ff: bool = True
    bias: bool = False
    dropout: float = 0.1
    W_qk_mode: str = 'none'
    W_v_mode: str = 'none'
    W_o_mode: str = 'sum'
    use_W_LR: bool = True
    use_W_N: bool = True
    kernel_function: str = 'softmax'
    
    def __post_init__(self):
        self.d_attn = self.d_attn or self.d_embed
        self.d_ff = self.d_ff or self.d_embed * 4
        assert self.W_qk_mode in ['none', 'diag', 'linear', 'diag_shared', 'linear_shared']
        assert self.W_v_mode in ['none', 'diag', 'linear']
        assert self.W_o_mode in ['sum', 'proj']
        assert self.kernel_function in ['softmax', 'linear', 'rbf', 'laplacian']