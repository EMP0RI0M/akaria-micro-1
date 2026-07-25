import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.fluxvm.divergence import FluxVMDivergence

class FluxVMLatentAdapter(nn.Module):
    """
    Differentiable PyTorch-FluxVM Latent Adapter implementing the
    Memory-Augmented Adaptive Control (MAAC) loop.
    Adapted from CI-Lang SwarmManager control law.
    """
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        
        # Made parameters trainable so the FluxVM optimizer bucket functions correctly
        self.tau = nn.Parameter(torch.tensor(cfg.tau, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(cfg.alpha, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(cfg.gamma, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(cfg.k, dtype=torch.float32))
        
        self.mode = cfg.flux_mode
        self.divergence_calc = FluxVMDivergence()
        
        self.s = 15.0 # Sigmoid steepness

    def forward(self, Y, loop_step, prev_memory=None, prev_D_global=0.0):
        # Y is (B, L, S, D)
        B = Y.shape[0]
        device = Y.device

        if prev_memory is None:
            prev_memory = torch.zeros(B, device=device)

        if self.mode == "OFF":
            return Y, prev_memory, {}

        # 1. Measure Divergence
        primary_D, metrics = self.divergence_calc(Y, prev_D_global)
        
        # We need a batch-wise primary signal for the control law
        # If primary_D is a scalar (averaged over batch), we expand it for the batch
        # To make it per-batch element, we compute it manually here:
        if Y.dim() == 4:
            stream_mean = Y.mean(dim=2, keepdim=True)
            batch_D = torch.mean((Y - stream_mean) ** 2, dim=(1,2,3))
        else:
            seq_mean = Y.mean(dim=1, keepdim=True)
            batch_D = torch.mean((Y - seq_mean) ** 2, dim=(1,2))

        # 2. Instability detection
        instability_soft = torch.sigmoid(self.s * (batch_D - self.tau))
        
        if not self.training:
            instability_hard = (batch_D > self.tau).float()
            instability = instability_hard + (instability_soft - instability_soft.detach())
        else:
            instability = instability_soft

        # 3. Memory Update
        current_memory = self.gamma * prev_memory + self.alpha * instability
        
        # 4. CHM Differentiable Adaptation of CI-Lang equation
        # beta_t = exp(-k * M_t). When M_t ≈ 0, beta_t ≈ 1.
        beta = torch.exp(-self.k * current_memory)
        
        metrics["M"] = current_memory
        metrics["beta"] = beta

        if self.mode == "CONTROL":
            if Y.dim() == 4:
                beta_broadcast = beta.view(B, 1, 1, 1)
                centroid = Y.mean(dim=(1, 2), keepdim=True)
            else:
                beta_broadcast = beta.view(B, 1, 1)
                centroid = Y.mean(dim=1, keepdim=True)
                
            Y_damped = beta_broadcast * Y + (1.0 - beta_broadcast) * centroid
        else:
            Y_damped = Y

        return Y_damped, current_memory, metrics
