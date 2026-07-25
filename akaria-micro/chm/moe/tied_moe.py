import sys
import os
import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.dense_ffn import DenseSwiGLU

# For Kaggle we assume looped-moe is installed via `pip install -e` or in PYTHONPATH
try:
    from model import MoELayer
except ImportError:
    # Fallback for local sandbox testing structure
    sys.path.append(os.path.abspath('/root/looped-moe'))
    from model import MoELayer

class TiedMoE(nn.Module):
    """
    Wraps looped-moe's highly optimized MoELayer.
    Ensures that for configs B and C, it falls back to a dense SwiGLU FFN.
    """
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        self.cfg = cfg
        
        if cfg.n_experts == 1:
            self.layer = DenseSwiGLU(cfg)
        else:
            # We use a dummy looped-moe Config object just to initialize its MoELayer
            class DummyConfig:
                d_model = cfg.d_model
                d_ff = cfg.expert_hidden_dim
                n_experts = cfg.n_experts
                experts_per_token = cfg.experts_per_token
                has_per_loop_routers = False
            
            self.layer = MoELayer(DummyConfig())
            
    def forward(self, x: torch.Tensor, loop_idx: int = 0):
        if self.cfg.n_experts == 1:
            return self.layer(x)
        else:
            # looped-moe MoELayer expects (x, loop_idx) and returns x
            # Since we set has_per_loop_routers=False, it uses the shared router
            out = self.layer(x, loop_idx)
            return out
