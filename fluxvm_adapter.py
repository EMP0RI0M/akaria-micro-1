import torch
import torch.nn as nn
import torch.nn.functional as F

class FluxVMLatentAdapter(nn.Module):
    """
    Differentiable PyTorch-FluxVM Latent Adapter implementing the
    Memory-Augmented Adaptive Control (MAAC) loop for deeply recurrent Transformers.
    """
    def __init__(self, hidden_dim, tau=1.5, alpha=0.1, gamma=0.9, k=2.5, mode="OBSERVE"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("k", torch.tensor(k))
        
        # mode can be "OFF", "OBSERVE", or "CONTROL"
        self.mode = mode

    def compute_divergence(self, H):
        centroid = H.mean(dim=1, keepdim=True)
        div = torch.mean((H - centroid) ** 2, dim=(1, 2))
        return div

    def forward(self, Y, loop_step, prev_memory=None):
        if self.mode == "OFF":
            return Y, prev_memory, torch.zeros(Y.shape[0], device=Y.device), torch.ones(Y.shape[0], device=Y.device)
            
        B, n_streams, L, C = Y.shape
        device = Y.device

        if prev_memory is None:
            prev_memory = torch.zeros(B, device=device)

        Y_mean = Y.mean(dim=1)
        div = self.compute_divergence(Y_mean)
        
        instability_soft = torch.sigmoid(15.0 * (div - self.tau))
        if not self.training:
            instability_hard = (div > self.tau).float()
            instability = instability_hard + (instability_soft - instability_soft.detach())
        else:
            instability = instability_soft

        current_memory = self.gamma * prev_memory + self.alpha * instability
        beta = torch.exp(-self.k * (1.0 + current_memory))
        
        if self.mode == "CONTROL":
            beta_broadcast = beta.view(B, 1, 1, 1)
            centroid = Y.mean(dim=2, keepdim=True)
            Y_damped = beta_broadcast * Y + (1.0 - beta_broadcast) * centroid
        else:
            # OBSERVE mode: return Y untouched
            Y_damped = Y

        return Y_damped, current_memory, div, beta
