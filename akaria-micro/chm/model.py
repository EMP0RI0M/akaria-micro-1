import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.normalization import RMSNorm
from chm.attention import CausalSelfAttention
from chm.moe.tied_moe import TiedMoE
from chm.recurrent.core import RecurrentCore
from chm.hyperloop.hyper_connections import HyperConnections
from chm.fluxvm.adapter import FluxVMLatentAdapter
from chm.init_utils import apply_weight_init

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
        if cfg.flux_mode == "E1":
            flux_adapter = FluxVMLatentAdapter(cfg)
        elif cfg.flux_mode in ["E0", "E2", "E3", "E4", "E5"]:
            from chm.fluxvm.fluxvm_v2 import FluxVMControllerV2
            flux_adapter = FluxVMControllerV2(
                tau=cfg.tau,
                alpha=cfg.alpha,
                gamma_P=cfg.gamma_P,
                gamma_I=cfg.gamma_I,
                gamma_D=cfg.gamma_D,
                buffer_size=cfg.buffer_size,
                K=cfg.K
            )
        elif cfg.flux_mode != "OFF":
            flux_adapter = FluxVMLatentAdapter(cfg)
        else:
            flux_adapter = None
        
        self.core = RecurrentCore(cfg, shared_block, hyper_connections, flux_adapter)
        
        self.coda = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.coda_layers)])
        
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Initialize weights
        num_res_layers = cfg.prelude_layers + cfg.core_loops + cfg.coda_layers
        apply_weight_init(self, std=0.02, num_residual_layers=num_res_layers)
        
        # Tie weights after init
        self.lm_head.weight = self.tok_emb.weight
        
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
