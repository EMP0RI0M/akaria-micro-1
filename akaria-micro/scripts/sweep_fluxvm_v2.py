import torch
import torch.nn as nn
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE

def run_v2_ablation(cfg: CHMConfig, batch_size=2, seq_len=32, steps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    model = ControlledHyperloopMoE(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    
    start_time = time.time()
    for step in range(steps):
        optimizer.zero_grad()
        logits, telemetry = model(x)
        lm_loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
        
        loss = lm_loss
        L_b_val = 0.0
        if "L_barrier" in telemetry and telemetry["L_barrier"]:
            L_b = torch.stack(telemetry["L_barrier"]).sum()
            loss = loss + L_b
            L_b_val = L_b.item()
            
        loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step == steps - 1:
            print(f"--- FluxMode: {cfg.flux_mode} | lambda_barrier: {cfg.lambda_barrier} ---")
            print(f"Total Params: {total_params / 1e6:.2f}M")
            print(f"LM Loss: {lm_loss.item():.4f} | Barrier Loss: {L_b_val:.4f} | Total Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            if "barrier_pass" in telemetry and telemetry["barrier_pass"]:
                bp_mean = sum(telemetry["barrier_pass"]) / len(telemetry["barrier_pass"])
                print(f"Barrier Pass Rate: {bp_mean:.4f}")
            
            # Print detailed telemetry for the final step
            keys_to_print = ["D_t", "D_prev", "delta_D", "M_t", "historical_M", "g_D", "g_M", "G_t", 
                             "P_term", "I_term", "D_term", "I_t", "V_before", "V_after", "delta_V"]
            for k in keys_to_print:
                if k in telemetry and telemetry[k]:
                    print(f"  {k}: {[f'{v:.4f}' for v in telemetry[k]]}")
            
    throughput = (batch_size * seq_len * steps) / (time.time() - start_time)
    print(f"Throughput: {throughput:.2f} tok/s")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    print("=== Sweeping FluxVM V2 Controller ===")
    
    configs = [
        ("E0", 0.0), # Observe
        ("E1", 0.0), # Old exponential controller
        ("E2", 0.0), # P/D + threshold
        ("E3", 0.0), # MAAC/PID + memory trigger
        ("E4", 0.0), # MAAC/PID + fixed-K barrier (no loss)
        ("E4", 0.1), # MAAC/PID + fixed-K barrier (with loss)
    ]
    
    for flux, l_bar in configs:
        cfg = CHMConfig(config_type="E", flux_mode=flux, lambda_barrier=l_bar)
        run_v2_ablation(cfg)
