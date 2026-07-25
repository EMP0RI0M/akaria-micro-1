import torch
import torch.nn as nn
from chm.config import CHMConfig

class HyperConnections(nn.Module):
    def __init__(self, cfg: CHMConfig):
        super().__init__()
        self.n_streams = cfg.n_streams
        self.d_model = cfg.d_model
        
        self.W_pre = nn.Linear(cfg.d_model * cfg.n_streams, cfg.n_streams)
        self.b_pre = nn.Parameter(torch.zeros(cfg.n_streams))
        
        self.W_post = nn.Linear(cfg.d_model, cfg.d_model * cfg.n_streams)
        
        self.W_res = nn.Linear(cfg.d_model * cfg.n_streams, cfg.n_streams)
        self.b_res = nn.Parameter(torch.zeros(cfg.n_streams))

    def pre_transform(self, Y: torch.Tensor):
        # Y is (B, L, S, D)
        B, L, S, D = Y.shape
        Z_prev = Y.view(B, L, -1)
        
        H_pre = torch.sigmoid(self.W_pre(Z_prev) + self.b_pre) # (B, L, S)
        X_attn = (Y * H_pre.unsqueeze(-1)).sum(dim=2) # (B, L, D)
        return X_attn, Y

    def post_transform(self, Y: torch.Tensor, H_tilde: torch.Tensor):
        B, L, S, D = Y.shape
        Z_prev = Y.view(B, L, -1)
        
        H_res = torch.sigmoid(self.W_res(Z_prev) + self.b_res) # (B, L, S)
        H_post = self.W_post(H_tilde).view(B, L, S, D) # (B, L, S, D)
        
        Y_new = Y * H_res.unsqueeze(-1) + H_post
        return Y_new
