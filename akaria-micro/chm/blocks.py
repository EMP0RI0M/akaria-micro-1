import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.normalization import RMSNorm
from chm.attention import CausalSelfAttention
from chm.moe.tied_moe import TiedMoE

class TransformerBlock(nn.Module):
    """A standard Transformer block, usable as Prelude, Coda, or the Shared Core."""
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = TiedMoE(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
