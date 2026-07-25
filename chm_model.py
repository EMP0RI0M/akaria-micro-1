import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath('/root/looped-moe'))
sys.path.append(os.path.abspath('/root/OpenMythos'))

from model import ModelConfig, TransformerBlock, RMSNorm, LoopedMoETransformer
from fluxvm_adapter import FluxVMLatentAdapter

class DenseSwiGLU(nn.Module):
    """Genuine standard dense SwiGLU FFN to bypass MoE routers for Configs B and C."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x, loop_idx=0):
        # Emulate the return signature of MoELayer: (output, router_logits)
        # We return None for router_logits
        return self.w2(F.silu(self.w1(x)) * self.w3(x)), None

class CHMConfig(ModelConfig):
    config_type: str = "E"
    n_streams: int = 4
    core_loops: int = 10
    prelude_layers: int = 2
    coda_layers: int = 2
    flux_mode: str = "OBSERVE"
    
    # FluxVM MAAC parameters
    tau: float = 1.0
    alpha: float = 0.1
    gamma: float = 0.9
    k: float = 2.5
    
    def __post_init__(self):
        if self.config_type == "A":
            self.topology = [((self.prelude_layers + self.core_loops + self.coda_layers), 1)]
            self.n_streams = 1
            self.flux_mode = "OFF"
        elif self.config_type == "B":
            self.topology = [(self.prelude_layers, 1), (1, self.core_loops), (self.coda_layers, 1)]
            self.n_streams = 1
            self.flux_mode = "OFF"
        elif self.config_type == "C":
            self.topology = [(self.prelude_layers, 1), (1, self.core_loops), (self.coda_layers, 1)]
            self.flux_mode = "OFF"
        elif self.config_type == "D":
            self.topology = [(self.prelude_layers, 1), (1, self.core_loops), (self.coda_layers, 1)]
            self.flux_mode = "OFF"
        elif self.config_type == "E":
            self.topology = [(self.prelude_layers, 1), (1, self.core_loops), (self.coda_layers, 1)]
            if self.flux_mode == "OFF": 
                self.flux_mode = "OBSERVE"

class RecurrentCore(nn.Module):
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        self.cfg = cfg
        self.n_streams = cfg.n_streams
        self.d_model = cfg.d_model
        
        self.tied_block = TransformerBlock(cfg, n_loops_for_group=cfg.core_loops)
        
        # Override MoELayer with DenseSwiGLU for Configs B and C
        if cfg.config_type in ["B", "C"]:
            self.tied_block.moe = DenseSwiGLU(cfg.d_model, cfg.d_ff)
        
        if self.n_streams > 1:
            self.W_pre = nn.Linear(cfg.d_model * cfg.n_streams, cfg.n_streams)
            self.b_pre = nn.Parameter(torch.zeros(cfg.n_streams))
            self.W_post = nn.Linear(cfg.d_model, cfg.d_model * cfg.n_streams)
            self.W_res = nn.Linear(cfg.d_model * cfg.n_streams, cfg.n_streams)
            self.b_res = nn.Parameter(torch.zeros(cfg.n_streams))
        else:
            self.W_z = nn.Linear(cfg.d_model * 2, cfg.d_model)
            self.b_z = nn.Parameter(torch.full((cfg.d_model,), -2.0))
            
        self.loop_embeds = nn.Parameter(torch.randn(cfg.core_loops, cfg.d_model) * 0.02)
        
        self.flux_adapter = FluxVMLatentAdapter(
            hidden_dim=cfg.d_model, tau=cfg.tau, alpha=cfg.alpha, gamma=cfg.gamma, k=cfg.k, mode=cfg.flux_mode
        )

    def forward(self, H_in):
        B, L, C = H_in.shape
        Y = H_in.unsqueeze(2).expand(B, L, self.n_streams, C).clone() if self.n_streams > 1 else H_in
        
        prev_memory = None
        all_router_logits = []
        divergence_log, beta_log = [], []
        
        for t in range(self.cfg.core_loops):
            if self.n_streams > 1:
                Z_prev = Y.reshape(B, L, -1)
                H_pre = torch.sigmoid(self.W_pre(Z_prev) + self.b_pre)
                X_attn = (Y * H_pre.unsqueeze(-1)).sum(dim=2) + self.loop_embeds[t].view(1, 1, C)
            else:
                X_attn = Y + self.loop_embeds[t].view(1, 1, C)
                
            H_tilde, router_logits = self.tied_block(X_attn, loop_idx=t)
            if router_logits is not None:
                all_router_logits.append(router_logits)
            
            if self.n_streams > 1:
                H_res = torch.sigmoid(self.W_res(Z_prev) + self.b_res)
                H_post = self.W_post(H_tilde).view(B, L, self.n_streams, C)
                Y = Y * H_res.unsqueeze(-1) + H_post
                Y_perm = Y.permute(0, 2, 1, 3)
                Y_damped, prev_memory, div, beta = self.flux_adapter(Y_perm, t, prev_memory)
                Y = Y_damped.permute(0, 2, 1, 3)
            else:
                gate = torch.sigmoid(self.W_z(torch.cat([Y, H_tilde], dim=-1)) + self.b_z)
                Y = gate * Y + (1 - gate) * H_tilde
                Y_damped, prev_memory, div, beta = self.flux_adapter(Y.unsqueeze(1), t, prev_memory)
                Y = Y_damped.squeeze(1)
                
            divergence_log.append(div.mean().item() if div is not None else 0.0)
            beta_log.append(beta.mean().item() if beta is not None else 1.0)
            
        H_out = Y.mean(dim=2) if self.n_streams > 1 else Y
        return H_out, all_router_logits, divergence_log, beta_log

class ControlledHyperloopMoE(nn.Module):
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        self.cfg = cfg
        
        if cfg.config_type == "A":
            self.baseline = LoopedMoETransformer(cfg)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.drop = nn.Dropout(cfg.dropout)
            self.prelude = nn.ModuleList([TransformerBlock(cfg, n_loops_for_group=1) for _ in range(cfg.prelude_layers)])
            
            # Override Prelude/Coda to DenseSwiGLU for Config B/C
            if cfg.config_type in ["B", "C"]:
                for layer in self.prelude:
                    layer.moe = DenseSwiGLU(cfg.d_model, cfg.d_ff)
                    
            self.core = RecurrentCore(cfg)
            
            self.coda = nn.ModuleList([TransformerBlock(cfg, n_loops_for_group=1) for _ in range(cfg.coda_layers)])
            if cfg.config_type in ["B", "C"]:
                for layer in self.coda:
                    layer.moe = DenseSwiGLU(cfg.d_model, cfg.d_ff)
                    
            self.norm_out = RMSNorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight
        
    def forward(self, input_ids):
        if self.cfg.config_type == "A":
            return self.baseline(input_ids)[0], [], [], []
            
        x = self.drop(self.tok_emb(input_ids))
        all_router_logits = []
        
        for layer in self.prelude:
            x, logits = layer(x, loop_idx=0)
            if logits is not None: all_router_logits.append(logits)
            
        x, core_logits, div_log, beta_log = self.core(x)
        all_router_logits.extend(core_logits)
        
        for layer in self.coda:
            x, logits = layer(x, loop_idx=0)
            if logits is not None: all_router_logits.append(logits)
            
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        return logits, all_router_logits, div_log, beta_log
