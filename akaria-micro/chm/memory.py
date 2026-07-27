import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE

class MemoryRefinement(nn.Module):
    """
    G_theta: iterative memory refinement using shared weights.
    M_t^(k+1) = M_t^(k) + G_theta(M_t^(k), H_t)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.memory_proj = nn.Linear(d_model, d_model)
        self.context_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, M: torch.Tensor, H_ctx: torch.Tensor) -> torch.Tensor:
        # M: (B, num_mem_tokens, D)
        # H_ctx: (B, chunk_size, D)
        
        # Simple attention-free pooling of the context to keep it extremely cheap
        ctx_pool = H_ctx.mean(dim=1, keepdim=True) # (B, 1, D)
        
        # Refinement
        h = self.act(self.memory_proj(M) + self.context_proj(ctx_pool))
        return self.out_proj(h)


class E0Memory(nn.Module):
    def __init__(self, cfg: CHMConfig, num_mem_tokens=16, mem_refinement_steps=1):
        super().__init__()
        self.cfg = cfg
        self.num_mem_tokens = num_mem_tokens
        self.mem_refinement_steps = mem_refinement_steps
        
        # Core E0 model reused
        self.e0 = ControlledHyperloopMoE(cfg)
        
        # Initial persistent memory state
        self.M_0 = nn.Parameter(torch.randn(1, num_mem_tokens, cfg.d_model) * 0.02)
        
        # Learned write/update gate
        self.w_g = nn.Linear(cfg.d_model, 1)
        
        # Iterative memory refinement
        self.g_theta = MemoryRefinement(cfg.d_model)
        
    def get_initial_memory(self, batch_size: int, device: torch.device):
        return self.M_0.expand(batch_size, -1, -1).clone()

    def forward_chunk(self, input_ids: torch.Tensor, memory_state: torch.Tensor = None):
        """
        Forward pass for a SINGLE chunk of input_ids.
        input_ids: (B, chunk_size)
        memory_state: (B, num_mem_tokens, D)
        """
        B, chunk_size = input_ids.shape
        device = input_ids.device
        
        if memory_state is None:
            memory_state = self.get_initial_memory(B, device)
            
        chunk_emb = self.e0.drop(self.e0.tok_emb(input_ids)) 
        
        # Causal attention will allow chunk tokens to attend to the memory prefix.
        combined_emb = torch.cat([memory_state, chunk_emb], dim=1) 
        
        logits_full, telemetry, features_full = self.e0(inputs_embeds=combined_emb, return_features=True)
        
        # Slice out chunk outputs
        logits_chunk = logits_full[:, self.num_mem_tokens:, :]
        H_ctx = features_full[:, self.num_mem_tokens:, :]
        
        # Memory Update candidate
        M_k = memory_state
        for _ in range(self.mem_refinement_steps):
            M_k = M_k + self.g_theta(M_k, H_ctx)
        C_t = M_k
        
        # Write gate g_t
        g_t = torch.sigmoid(self.w_g(C_t)) 
        
        new_memory_state = (1 - g_t) * memory_state + g_t * C_t
        
        if "mem_gate" not in telemetry:
            telemetry["mem_gate"] = []
            telemetry["mem_update_mag"] = []
            
        telemetry["mem_gate"].append(g_t)
        telemetry["mem_update_mag"].append((new_memory_state - memory_state).norm(dim=-1))
        
        return logits_chunk, telemetry, new_memory_state

    def forward(self, input_ids: torch.Tensor, chunk_size: int = 256, detach_memory_every: int = 1, return_last_chunk_only: bool = False):
        """
        Process a long logical sequence by chunking it.
        Implements Truncated Backpropagation Through Time (TBPTT).
        detach_memory_every: number of chunks before detaching memory state to save VRAM.
        return_last_chunk_only: if True, only returns logits for the final chunk, freeing VRAM for earlier chunks if memory is detached.
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        memory_state = self.get_initial_memory(B, device)
        all_logits = []
        all_telemetry = {}
        
        for chunk_idx, start_idx in enumerate(range(0, L, chunk_size)):
            end_idx = min(start_idx + chunk_size, L)
            chunk_ids = input_ids[:, start_idx:end_idx]
            
            # Detach memory if needed (TBPTT)
            if detach_memory_every > 0 and chunk_idx > 0 and chunk_idx % detach_memory_every == 0:
                memory_state = memory_state.detach()
                
            logits_chunk, telemetry, memory_state = self.forward_chunk(chunk_ids, memory_state)
            
            if not return_last_chunk_only:
                # If we are doing TBPTT but keeping all logits, we must detach early logits 
                # to actually free the graph, OR keep them attached if the user wants full gradients.
                # Standard causal LM requires gradients, but retaining all graphs will OOM.
                # For safety in this wrapper, we just append. To avoid OOM, use return_last_chunk_only=True
                all_logits.append(logits_chunk)
            else:
                all_logits = [logits_chunk] # Only keep the latest one
            
            # Accumulate telemetry
            for k, v in telemetry.items():
                if k not in all_telemetry:
                    all_telemetry[k] = []
                all_telemetry[k].extend(v)
                
        final_logits = torch.cat(all_logits, dim=1)
        return final_logits, all_telemetry
