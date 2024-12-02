import math

import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Attention heads
        self.W_q = nn.Parameter(torch.zeros(1, config.n_head, config.d_embed, config.d_embed))
        self.W_k = nn.Parameter(torch.zeros(1, config.n_head, config.d_embed, config.d_embed))
        self.W_v = nn.Parameter(torch.zeros(1, config.n_head, config.d_embed, config.d_embed))
        self.W_o = nn.Parameter(torch.zeros(1, config.n_head * config.d_embed, config.d_embed))
        
        # Layernorm and Dropout
        if config.use_ppe:
            self.ln_e = nn.LayerNorm(config.d_embed, bias=False)
            self.ln_p = nn.LayerNorm(config.d_embed, bias=False)
        else:
            self.ln_x = nn.LayerNorm(config.d_embed, bias=False)
        
        self.dropout_attn = nn.Dropout(config.dropout)
        self.dropout_output = nn.Dropout(config.dropout)
    
    def _init_weights(self):
        nn.init.normal_(self.W_q, std=0.02)
        nn.init.normal_(self.W_k, std=0.02)
        nn.init.normal_(self.W_v, std=0.02)
        nn.init.normal_(self.W_o, std=0.02 / math.sqrt(2 * self.config.n_layer))
    
    def forward(self, x, e, p):
        
        device = x.device
        B, S, _ = x.size()
        
        if self.config.use_ppe:
            e = self.ln_e(e).unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
            p = self.ln_p(p).unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
            Q = torch.matmul(p, self.W_q)
            K = torch.matmul(p, self.W_k)
            V = torch.matmul(e, self.W_v)
        else:
            x = self.ln_x(x).unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
            Q = torch.matmul(x, self.W_q)
            K = torch.matmul(x, self.W_k)
            V = torch.matmul(x, self.W_v)
        
        # Compute attention scores
        if self.attn_kernel_fn == 'softmax':
            attn_scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.config.d_embed), dim=-1)
        elif self.attn_kernel_fn == 'linear':
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.config.d_embed)
        elif self.attn_kernel_fn == 'rbf':
            attn_scores = torch.cdist(Q, K, p=2).pow(2).mul(-self.gamma).exp()
        elif self.attn_kernel_fn == 'laplacian':
            attn_scores = torch.cdist(Q, K, p=1).mul(-self.gamma).exp()
        
        # Add causal mask (if not next_target_only)
        if not self.config.nto:
            mask = torch.tril(torch.ones(S, S, device=device))
            mask = mask.bool()
            attn_bias = torch.zeros(S, S, device=device)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_scores += attn_bias
        
        # Apply dropout
        attn_scores = self.dropout_attn(attn_scores)
        
        # Compute attention output
        attn_output = torch.matmul(attn_scores, V)
        attn_output = torch.matmul(attn_output.transpose(1, 2).contiguous().view(B, S, -1), self.W_o)
        attn_output = self.dropout_output(attn_output)
        
        return attn_output

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.attn = Attention(config)
        
        if config.use_ff:
            self.ff = nn.Sequential(
                nn.LayerNorm(config.d_embed, bias=False),
                nn.Linear(config.d_embed, config.d_ff, bias=False),
                nn.GELU(),
                nn.Linear(config.d_ff, config.d_embed, bias=False),
                nn.Dropout(config.dropout)
            )
        self._init_weights()
        
    def _init_weights(self):
        if self.config.use_ff:
            nn.init.normal_(self.ff[1].weight, std=0.02)
            nn.init.normal_(self.ff[3].weight, std=0.02)

    def forward(self, x, e, p):
        # If using PPE, only use e for skip connection
        if self.config.use_ppe:
            x = e + self.attn(x, e, p)
        else:
            x = x + self.attn(x, e, p)
        if self.config.use_ff:
            x = x + self.ff(x)
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.name = f'GPT_{config.d_embed}D_{config.n_layer}L_{config.n_head}H_K={config.attn_kernel_fn}'
        
        if config.use_ff:
            self.name += '_FF'
        if config.use_ppe:
            self.name += '_PPE'
        if config.use_nto:
            self.name += '_NTO'
        
        # Embedding
        self.W_e = nn.Embedding(config.vocab_size, config.d_embed)
        self.W_p = nn.Embedding(config.context_size, config.d_embed)
        self.drop_e = nn.Dropout(config.dropout)
        self.drop_p = nn.Dropout(config.dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        
        # LM Head
        self.ln_out = nn.LayerNorm(config.d_embed, bias=False)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.W_e.weight = self.lm_head.weight # Weight tying

        # Weight initialization
        self._init_weights()

        # Parameter Count
        self.n_params = sum(p.numel() for p in self.parameters()) - self.W_e.weight.numel() - self.W_p.weight.numel() # Exclude embedding weights
        self.n_params_formatted = f'{self.n_params/1e6:.2f}M'
        print(f'Initialized model {self.name} with {self.n_params_formatted} parameters')
        
    def _init_weights(self):
        torch.nn.init.normal_(self.W_e.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.W_p.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        
        device = x.device
        B, S = x.size()

        # Embeddings
        e = self.W_e(x)
        p = self.W_p(torch.arange(0, S, dtype=torch.long, device=device)).unsqueeze(0)
        e = self.drop_e(e)
        p = self.drop_p(p)
        x = e + p
        
        for block in self.blocks:
            x = block(x, e, p)
        
        x = self.ln_out(x)

        if targets is None:
            logits = self.lm_head(x)
            loss = None
        else:
            logits = self.lm_head(x)
            if self.config.use_nto: # Only predict the N+1 token
                logits = logits[:, -1, :]
                targets = targets[:, -1]
            targets = targets.contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    # Generates new tokens based on the most probable N+1th token at each step
    def generate(self, x, max_new_tokens=100, eos_token=None):
        
        for _ in range(max_new_tokens):
            logits, _ = self(x)
            x_next = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            x = torch.cat((x, x_next), dim=1)
            if eos_token is not None and x_next.item() == eos_token:
                break
            
        return x
    
    # Generates new tokens based on the most probable sequence of tokens
    def beam_search(self, x, max_new_tokens=100, num_beams=3, eos_token=None):
    
        beams = [{'x': x, 'score': 0, 'eos': False}]  # Initial beam
        
        for _ in range(max_new_tokens):
            
            new_sequences = []
            
            for beam in beams:
            
                # If EOS is already encountered, propagate the beam without changes
                if beam['eos']:
                    new_sequences.append(beam)
                    continue
                
                # Generate beam candidates
                logits, _ = self(beam['x'])
                topk = torch.topk(logits[:, -1, :], num_beams, dim=-1)
                
                for i in range(num_beams):
                    x_next = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
                    score = topk.values[0, i].item()
                    new_x = torch.cat((beam['x'], x_next), dim=1)
                    new_eos = eos_token is not None and x_next.item() == eos_token
                    new_sequences.append({
                        'x': new_x,
                        'score': beam['score'] + score,
                        'eos': new_eos
                    })
            
            # Select beam based on normalized score
            new_sequences.sort(key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1), reverse=True)
            beams = new_sequences[:num_beams]
            
            # Break early if all beams have encountered EOS
            if all(beam['eos'] for beam in beams):
                break
        
        most_probable_sequence = max(beams, key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1))
        return most_probable_sequence['x']