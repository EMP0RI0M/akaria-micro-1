import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.blocks import TransformerBlock
from chm.recurrent.loop_embeddings import LoopEmbeddings

class RecurrentCore(nn.Module):
    def __init__(self, cfg: CHMConfig, shared_block: nn.Module, hyper_connections=None, flux_adapter=None):
        super().__init__()
        self.cfg = cfg
        self.shared_block = shared_block
        self.loop_embeds = LoopEmbeddings(cfg)
        self.hyper_connections = hyper_connections
        self.flux_adapter = flux_adapter

    def forward(self, H_in: torch.Tensor):
        # H_in: (B, L, D) -> Y: (B, L, S, D)
        B, L, D = H_in.shape
        S = self.cfg.n_streams

        if self.hyper_connections is not None:
            Y = H_in.unsqueeze(2).expand(B, L, S, D).clone()
        else:
            Y = H_in

        prev_memory = None
        telemetry = {
            "D_token": [], "D_stream": [], "D_global": [],
            "delta_D": [], "M": [], "beta": []
        }

        for t in range(self.cfg.core_loops):
            emb = self.loop_embeds(t).view(1, 1, D)
            
            if self.hyper_connections is not None:
                X_attn, Y = self.hyper_connections.pre_transform(Y)
                X_attn = X_attn + emb
            else:
                X_attn = Y + emb

            # shared block execution
            H_tilde = self.shared_block(X_attn)

            if self.hyper_connections is not None:
                Y = self.hyper_connections.post_transform(Y, H_tilde)
            else:
                Y = H_tilde

            if self.flux_adapter is not None:
                from chm.fluxvm.fluxvm_v2 import FluxVMControllerV2
                if isinstance(self.flux_adapter, FluxVMControllerV2):
                    Y_damped, L_barrier, barrier_pass = self.flux_adapter(
                        Y,
                        ablation_mode=self.cfg.flux_mode,
                        lambda_barrier=self.cfg.lambda_barrier
                    )
                    Y = Y_damped
                    metrics = {"L_barrier": L_barrier, "barrier_pass": barrier_pass}
                else:
                    Y_damped, prev_memory, metrics = self.flux_adapter(Y, t, prev_memory)
                    Y = Y_damped
                
                for k, v in metrics.items():
                    if k not in telemetry:
                        telemetry[k] = []
                    if k == "L_barrier":
                        telemetry[k].append(v)
                    else:
                        telemetry[k].append(v.mean().item() if isinstance(v, torch.Tensor) else v)

        if self.hyper_connections is not None:
            H_out = Y.mean(dim=2)
        else:
            H_out = Y

        return H_out, telemetry
