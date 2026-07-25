import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm + self.eps)
        return x_normed.type_as(x) * self.weight
