import torch
import torch.nn as nn
import math

def init_weights(module: nn.Module, std: float = 0.02):
    """
    Standard GPT-2 style initialization.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

def apply_weight_init(model: nn.Module, std: float = 0.02, num_residual_layers: int = 1):
    model.apply(lambda m: init_weights(m, std))
    
    # Scale residual projections
    for name, p in model.named_parameters():
        if "out_proj.weight" in name or "down_proj.weight" in name:
            with torch.no_grad():
                p.mul_(1.0 / math.sqrt(2.0 * num_residual_layers))
