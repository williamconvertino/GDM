import math

import torch
from torch import nn
from torch.nn import functional as F

class GDStep(nn.Module):
    
    def __init__(self, W_e):
        super().__init__()
        
        self.W_e = W_e

    def forward(self, e, p, f_k):
        
        x_i = p
        print(x_i.shape)

class PGD(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.name = f'PGD_({config.d_embed}D)_({config.n_layer}L)_({config.n_head}H)'
        
        # Components
        self.W_e = nn.Embedding(config.vocab_size, config.d_embed)
        self.W_p = nn.Embedding(config.context_size + 1, config.d_embed)
        
        self.steps = nn.ModuleList([GDStep(self.W_e) for _ in range(config.n_layer)])
        
        # LM Head
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.W_e.weight = self.lm_head.weight # Weight tying, required by GD model

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.W_e.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        
        device = idx.device
        B, S = idx.size()
        assert S <= self.config.context_size, f"Cannot forward sequence of length {S}, context size is only {self.context_size}"
        
        pos = torch.arange(0, S + 1, dtype=torch.long, device=device)

        # Embeddings

        e = self.W_e(idx) # token embeddings of shape (B, S, d_embed)
        p = self.W_p(pos).repeat(B, 1, 1) # position embeddings of shape (B, S + 1, d_embed)
    
        f_k = torch.zeros_like(e) # initial state of the model
    
        # Steps
    
        for step in self.steps:
            f_k = step(e, p, f_k)

        # LM Head Outputs + Loss

        if targets is not None:
            logits = self.lm_head(x)
            targets = targets.contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(f_k[:, [-1], :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_size
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx