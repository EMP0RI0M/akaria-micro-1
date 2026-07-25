import torch
import torch.nn as nn
from chm.config import CHMConfig

class LoopEmbeddings(nn.Module):
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(cfg.core_loops, cfg.d_model) * 0.02)
        
    def forward(self, t: int) -> torch.Tensor:
        return self.embeddings[t]
