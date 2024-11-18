"""
A (heavily) modified version of karpathy's nanoGPT model (https://github.com/karpathy/nanoGPT)
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

class GDAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.n_head = config.n_head
        self.d_embed = config.d_embed
        self.context_size = config.context_size
        
        self.WQK_mode = config.WQK_mode
        self.WV_mode = config.WV_mode
        self.WO_mode = config.WO_mode
        
        if self.WQK_mode == 'full':
            self.W_qk = nn.Parameter(torch.zeros(1, self.n_head, self.d_embed, self.d_embed))
            nn.init.normal_(self.W_qk, mean=0.0, std=0.2)
        elif self.WQK_mode == 'diag':
            self.qk_diag_values = nn.Parameter(torch.zeros(self.n_head, self.d_embed))
            nn.init.normal_(self.qk_diag_values, mean=0.0, std=0.2)
            
        W_N = torch.diag_embed(torch.tensor([1.0 / (i + 1) for i in range(self.context_size)])).unsqueeze(0).unsqueeze(0)
        self.register_buffer('W_N', W_N)
        
        self.W_LR = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        nn.init.normal_(self.W_LR, mean=0.0, std=0.01)
        
        if self.WO_mode == 'proj':
            self.wo_proj = nn.Linear(self.n_head * self.d_embed, self.d_embed)
        
    def forward(self, e, p):
        B, S, D = e.size()

        Q = p[:, 1:, :].unsqueeze(1).repeat(1, self.n_head, 1, 1)
        K = p[:, :-1, :].unsqueeze(1).repeat(1, self.n_head, 1, 1)
        V = e.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        if self.WQK_mode == 'full':
            Q = Q @ self.W_qk
            K = K @ self.W_qk
            V = V
        elif self.WQK_mode == 'diag':
            W_qk = torch.diag_embed(self.qk_diag_values).unsqueeze(0)
            Q = Q @ W_qk
            K = K @ W_qk
            V = V # No need for a W_v matrix
        
        mask = torch.tril(torch.ones(S, S, device=e.device))
        mask = mask.bool()
        
        attn_bias = torch.zeros(S, S, device=e.device)
        attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        attn_weight = Q @ K.transpose(-2, -1)
        attn_weight += attn_bias
        attn_weight = F.softmax(attn_weight, dim=-1)
        
        y = attn_weight @ V
               
        y = self.W_N[:, :, :S, :S] @ y
        y = y * self.W_LR
        
        if self.WO_mode == 'proj':
            y = y.view(B, self.n_head * S, D).reshape(B, S, self.n_head * D)
            y = self.wo_proj(y)
            y = y.view(B, S, D)
            print(y)
        else:
            y = torch.sum(y, dim=1)
        
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.use_ff = config.use_ff
        
        self.attn = GDAttention(config)
        
        if self.use_ff:
            self.ln_mlp = nn.LayerNorm(config.d_embed, bias=config.bias)
            self.mlp = nn.Sequential(
                nn.Linear(config.d_embed, config.d_ff, bias=config.bias),
                nn.GELU(),
                nn.Linear(config.d_ff, config.d_embed, bias=config.bias),
                nn.Dropout(config.dropout)
            )

    def forward(self, e, p):
        
        x = self.attn(e, p)
        
        if self.use_ff:
            x = x + self.mlp(self.ln_mlp(x))
        
        return x

class gdGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        # self.name = f'gdGPT_{config.n_head}H_{config.n_layer}L_{config.d_embed}D'
        self.name = f'gdGPT'
        
        self.name += f'_WQK={config.WQK_mode}'
        self.name += f'_WV={config.WV_mode}'
        self.name += f'_WO={config.WO_mode}'
        
        # Transformer Components
        self.wte = nn.Embedding(config.vocab_size, config.d_embed)
        self.wpe = nn.Embedding(config.context_size + 1, config.d_embed) # Need a positional vector for the N+1th token
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # LM Head
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight # Weight tying

        # Weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        
        device = idx.device
        B, S = idx.size()
        assert S <= self.config.context_size, f"Cannot forward sequence of length {S}, context size is only {self.context_size}"
        
        pos = torch.arange(0, S + 1, dtype=torch.long, device=device)

        e = self.wte(idx) # token embeddings of shape (B, S, d_embed)
        p = self.wpe(pos).repeat(B, 1, 1) # position embeddings of shape (B, S + 1, d_embed)
    
        for block in self.blocks:
            x = block(e, p)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            targets = targets.contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
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
    
    def beam_search_generate(self, x, tokenizer, max_length=100, beam_width=5, max_ngrams=None, return_all_beams=False, only_return_new_tokens=False):
        
        if isinstance(x, str):
            x = tokenizer(x, return_tensors='pt')['input_ids']
        
        original_sequence_length = x.size(1)
        
        beams = [{'sequence': x, 'score': 0, 'eos': False, 'length': 0}]
        
        self.eval()
        with torch.no_grad():
            while not all([beam['eos'] for beam in beams]) and max([beam['length'] for beam in beams]) < max_length:
                updated_beams = []
                for beam in beams:
                    if beam['eos']:
                        updated_beams.append(beam)
                        continue
                logits, _ = self.forward(beam['sequence'])
                logits = logits[0, 0, :]
                top_k_logits, top_k_tokens = torch.topk(logits, beam_width)
                for i in range(beam_width):
                    new_sequence = torch.cat([beam['sequence'], top_k_tokens[i].unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = beam['score'] + top_k_logits[i].item()
                    new_eos = top_k_tokens[i].item() == tokenizer.eos_token_id
                    new_length = beam['length'] + 1
                    updated_beams.append({'sequence': new_sequence, 'score': new_score, 'eos': new_eos, 'length': new_length})
                
                if max_ngrams is not None:
                    valid_beams = []
                
                    for beam in updated_beams:
                        sequence = beam['sequence'][0].tolist()
                        ngrams = [tuple(sequence[i:i+max_ngrams]) for i in range(len(sequence)-max_ngrams+1)]
                        if len(ngrams) == len(set(ngrams)):
                            valid_beams.append(beam)
                    
                    updated_beams = valid_beams
                
                beams = sorted(updated_beams, key=lambda x: x['score'] / x['length'], reverse=True)[:beam_width]
            
        if return_all_beams:
            return [tokenizer.decode(beam['sequence'][0].tolist()) for beam in beams]
        
        output = beams[0]['sequence'][0].tolist()
        
        if only_return_new_tokens:
            output = output[original_sequence_length:]

        return tokenizer.decode(output)