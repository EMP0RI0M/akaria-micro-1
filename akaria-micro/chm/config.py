import dataclasses
from typing import Optional

@dataclasses.dataclass
class CHMConfig:
    # Model dimensions
    vocab_size: int = 16384
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    d_ff: int = 1024
    dropout: float = 0.0

    # Topology
    prelude_layers: int = 1
    core_loops: int = 5
    coda_layers: int = 1

    # Hyperloop
    n_streams: int = 1

    # MoE
    n_experts: int = 1
    experts_per_token: int = 1
    expert_hidden_dim: int = 1024

    # FluxVM
    flux_mode: str = "OFF"
    tau: float = 1.0
    alpha: float = 0.9
    gamma: float = 0.9
    k: float = 2.5
    
    # FluxVM V2 Parameters
    gamma_P: float = 0.1
    gamma_I: float = 0.01
    gamma_D: float = 0.05
    buffer_size: int = 20
    K: int = 2
    lambda_barrier: float = 0.1

    # Sequence
    sequence_length: int = 512

    # Configuration type: A, A-MoE, B, C, D, E
    config_type: str = "A"

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.config_type == "A":
            self.n_streams = 1
            self.n_experts = 1
            self.flux_mode = "OFF"
        elif self.config_type == "B":
            self.n_streams = 1
            self.n_experts = 1
            self.flux_mode = "OFF"
        elif self.config_type == "C":
            self.n_streams = 4
            self.n_experts = 1
            self.flux_mode = "OFF"
        elif self.config_type == "D":
            self.n_streams = 4
            self.n_experts = 8
            self.experts_per_token = 2
            self.flux_mode = "OFF"
        elif self.config_type == "E":
            self.n_streams = 4
            self.n_experts = 8
            self.experts_per_token = 2
            if self.flux_mode == "OFF":
                self.flux_mode = "OBSERVE"
