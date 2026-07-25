import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from chm.config import CHMConfig

def precompute_rope_freqs(d_head: int, seq_len: int, base: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb

def apply_rope(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return (x * freqs_cos) + (x_rotated * freqs_sin)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = self.d_model // self.n_heads
        
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(self.n_heads * self.d_head, cfg.d_model, bias=False)
        
        self.register_buffer("freqs_cos", precompute_rope_freqs(self.d_head, cfg.sequence_length).cos())
        self.register_buffer("freqs_sin", precompute_rope_freqs(self.d_head, cfg.sequence_length).sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.wq(x).view(B, L, self.n_heads, self.d_head)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.d_head)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.d_head)
        
        q = apply_rope(q, self.freqs_cos[:L].unsqueeze(0).unsqueeze(2), self.freqs_sin[:L].unsqueeze(0).unsqueeze(2))
        k = apply_rope(k, self.freqs_cos[:L].unsqueeze(0).unsqueeze(2), self.freqs_sin[:L].unsqueeze(0).unsqueeze(2))
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA repeat
        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = torch.repeat_interleave(k, num_repeat, dim=1)
            v = torch.repeat_interleave(v, num_repeat, dim=1)
            
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        
        return self.wo(out)
