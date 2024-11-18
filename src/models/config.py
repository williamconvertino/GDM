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
    use_ppe_encoding: bool = False # For GPT only
    WQK_mode: str = 'none'
    WV_mode: str = 'none'
    WO_mode: str = 'sum'
    
    def __post_init__(self):
        self.d_attn = self.d_attn or self.d_embed
        self.d_ff = self.d_ff or self.d_embed * 4
        assert self.WQK_mode in ['none', 'diag', 'full']
        assert self.WV_mode in ['none', 'diag', 'full']
        assert self.WO_mode in ['sum', 'proj']