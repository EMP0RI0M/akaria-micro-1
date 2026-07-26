import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.blocks import TransformerBlock
from chm.init_utils import apply_weight_init

class StandardTransformer(nn.Module):
    """
    A standard non-recurrent Transformer baseline.
    Configured to match the exact parameter count of the CHM model.
    """
    def __init__(self, cfg: CHMConfig, num_layers: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        
        # We use a config with config_type="A" (dense FFN) for the layers
        import copy
        layer_cfg = copy.deepcopy(cfg)
        layer_cfg.config_type = "A"
        layer_cfg.n_streams = 1
        layer_cfg.n_experts = 1
        
        self.layers = nn.ModuleList([
            TransformerBlock(layer_cfg) for _ in range(num_layers)
        ])
        
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        apply_weight_init(self, std=0.02, num_residual_layers=num_layers)
        
        # Tie weights
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids):
        x = self.drop(self.tok_emb(input_ids))
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        return logits, {} # Empty telemetry for API compatibility
