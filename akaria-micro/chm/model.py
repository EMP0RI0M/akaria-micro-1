import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.normalization import RMSNorm
from chm.attention import CausalSelfAttention
from chm.moe.tied_moe import TiedMoE
from chm.recurrent.core import RecurrentCore
from chm.hyperloop.hyper_connections import HyperConnections
from chm.fluxvm.adapter import FluxVMLatentAdapter

from chm.blocks import TransformerBlock

class ControlledHyperloopMoE(nn.Module):
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        self.cfg = cfg
        
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.prelude_layers)])
        
        # Shared core components
        shared_block = TransformerBlock(cfg)
        hyper_connections = HyperConnections(cfg) if cfg.n_streams > 1 else None
        flux_adapter = FluxVMLatentAdapter(cfg) if cfg.flux_mode != "OFF" else None
        
        self.core = RecurrentCore(cfg, shared_block, hyper_connections, flux_adapter)
        
        self.coda = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.coda_layers)])
        
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight # Weight tying
        
    def forward(self, input_ids: torch.Tensor):
        x = self.drop(self.tok_emb(input_ids))
        
        for layer in self.prelude:
            x = layer(x)
            
        x, telemetry = self.core(x)
        
        for layer in self.coda:
            x = layer(x)
            
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        return logits, telemetry
