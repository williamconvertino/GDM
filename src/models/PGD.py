import math

import torch
from torch import nn
from torch.nn import functional as F

class PGD(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.name = f'PGD_({config.d_embed}D)_({config.n_layer}L)_({config.n_head}H)_(kernel={config.kernel_function})'
        
        # Params
        self.n_head = config.n_head
        self.d_embed = config.d_embed
        self.context_size = config.context_size
        self.n_layer = config.n_layer
        self.kernel_function = config.kernel_function
        
        if self.kernel_function == 'rbf' or self.kernel_function == 'laplacian':
            self.gamma = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
            nn.init.constant_(self.gamma, 1.0)
        
        # Components
        self.W_e = nn.Embedding(config.vocab_size, config.d_embed)
        self.W_p = nn.Embedding(config.context_size + 1, config.d_embed)
        
        self.W_k = nn.Parameter(torch.zeros(1, self.n_head, config.d_embed, config.d_embed))
        self.W_q = nn.Parameter(torch.zeros(1, self.n_head, config.d_embed, config.d_embed))
        self.W_v = nn.Parameter(torch.zeros(1, self.n_head, config.d_embed, config.d_embed))

        self.A_LR = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        self.B_LR = nn.Parameter(torch.zeros(1, 1, 1))
        
        nn.init.normal_(self.W_e.weight, std=0.02)
        nn.init.normal_(self.W_p.weight, std=0.02)

        nn.init.normal_(self.W_k, std=0.02)
        nn.init.normal_(self.W_q, std=0.02)
        nn.init.normal_(self.W_v, std=0.02)
        
        nn.init.constant_(self.A_LR, 1.0)
        nn.init.constant_(self.B_LR, 0.01)
        
        # LM Head
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.W_e.weight = self.lm_head.weight # Weight tying, required by GD model

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.W_e.weight.numel()
        return n_params

    def gd_step(self, W_y_i, f_k, K):
        
        N = K.size(-1)
        temp = f_k[:, :N, :] @ self.W_e.weight.transpose(-2, -1)
        temp = torch.clamp(temp, -10, 10)
        exp_f_k_W_e = torch.exp(temp) # shape (B, S + 1, vocab_size)
        E_W_c = (exp_f_k_W_e @ self.W_e.weight) / (torch.sum(exp_f_k_W_e, dim=-1).unsqueeze(-1) + 1e-8) # shape (B, S + 1, d_embed)
        
        diff = W_y_i - E_W_c
        
        # W_v = torch.diag_embed(self.W_v_diag_values).unsqueeze(0)
        
        V = diff.unsqueeze(1).repeat(1, self.n_head, 1, 1) @ self.W_v
        delta_A = K @ V # shape (B, n_head, S + 1, d_embed)
        
        delta_A = delta_A * self.A_LR
        delta_B = (diff * self.B_LR).unsqueeze(1)

        delta_f_k = delta_A.sum(dim=1) + delta_B.sum(dim=2) # shape (B, S + 1, d_embed)
        delta_f_k = delta_f_k / N
        
        return f_k + delta_f_k
             
    def forward(self, idx, targets=None):
        
        device = idx.device
        B, S = idx.size()
        assert S <= self.config.context_size, f"Cannot forward sequence of length {S}, context size is only {self.context_size}"
        
        pos = torch.arange(0, S + 1, dtype=torch.long, device=device)

        # Embeddings

        e = self.W_e(idx) # token embeddings of shape (B, S, d_embed)
        p = self.W_p(pos).repeat(B, 1, 1) # position embeddings of shape (B, S + 1, d_embed)

        W_y_i = e
    
        x = p.unsqueeze(1).repeat(1, self.n_head, 1, 1) # shape (B, n_head, S + 1, d_embed)
        
        x_i = x @ self.W_k
        x_j = x @ self.W_q
        
        if self.kernel_function == 'linear':
            K = x_j @ x_i.transpose(-2, -1)
        elif self.kernel_function == 'softmax':
            K = F.softmax(x_j @ x_i.transpose(-2, -1), dim=-1)
        elif self.kernel_function == 'rbf':
            dist_sq = torch.cdist(x_j, x_i, p=2).pow(2)
            K = torch.exp(-self.gamma * dist_sq)
        elif self.kernel_function == 'laplacian':
            dist = torch.cdist(x_j, x_i, p=1)
            K = torch.exp(-self.gamma * dist)
            
        K = K[:, :, :, :-1]
        
        # K = K / (K.sum(dim=-1).unsqueeze(-1) + 1e-8)
            
        f_k = torch.zeros_like(p) # initial state of the model
        
        # Steps
    
        for _ in range(self.n_layer):
            f_k = self.gd_step(W_y_i, f_k, K)

        gd_output = f_k[:, -1, :]

        # LM Head Outputs + Loss

        if targets is not None:
            logits = self.lm_head(gd_output)
            targets = targets[:, -1]
            targets = targets.contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(gd_output)
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