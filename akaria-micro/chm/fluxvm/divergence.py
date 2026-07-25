import torch
import torch.nn as nn

class FluxVMDivergence(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Y: torch.Tensor, prev_D_global: float = 0.0):
        # Y is (B, L, S, D) or (B, L, D)
        has_streams = Y.dim() == 4
        
        metrics = {}
        
        # 1. Global activation dispersion (variance across all dimensions)
        D_global = torch.var(Y).item()
        metrics["D_global"] = D_global
        
        # 2. Delta divergence between iterations
        metrics["delta_D"] = D_global - prev_D_global
        
        # 3. Activation RMS
        metrics["activation_RMS"] = torch.sqrt(torch.mean(Y ** 2)).item()
        
        if has_streams:
            # Token/sequence divergence: variance across L, averaged over B, S, D
            seq_mean = Y.mean(dim=1, keepdim=True)
            D_token = torch.mean((Y - seq_mean) ** 2, dim=1).mean().item()
            
            # Cross-stream divergence: variance across S, averaged over B, L, D
            stream_mean = Y.mean(dim=2, keepdim=True)
            D_stream = torch.mean((Y - stream_mean) ** 2, dim=2).mean().item()
            
            metrics["D_token"] = D_token
            metrics["D_stream"] = D_stream
            
            # We use D_stream as the primary instability signal if streams exist
            primary_signal = D_stream
        else:
            seq_mean = Y.mean(dim=1, keepdim=True)
            D_token = torch.mean((Y - seq_mean) ** 2, dim=1).mean().item()
            
            metrics["D_token"] = D_token
            metrics["D_stream"] = 0.0
            
            primary_signal = D_token
            
        return primary_signal, metrics
