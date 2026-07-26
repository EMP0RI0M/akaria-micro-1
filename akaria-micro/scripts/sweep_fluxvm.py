import torch
import torch.nn as nn
import time
from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE

def run_sweep_step(k, alpha, batch_size=2, seq_len=32, steps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42) # Ensure exact deterministic batches/initialization
    
    # Base Config E with CONTROL
    cfg = CHMConfig(
        config_type="E", 
        flux_mode="CONTROL",
        k=k,
        alpha=alpha,
        gamma=0.9
    )
    model = ControlledHyperloopMoE(cfg).to(device)
    
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    
    for step in range(steps):
        optimizer.zero_grad()
        logits, telemetry = model(x)
        loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step == steps - 1:
            print(f"| k={k:<5} alpha={alpha:<4} | Loss: {loss.item():.4f} | Grad: {grad_norm.item():.2f} |")
            print(f"  D_t   : {[f'{d:.2f}' for d in telemetry['D_stream']]}")
            print(f"  M_t   : {[f'{m:.2f}' for m in telemetry['M']]}")
            print(f"  beta_t: {[f'{b:.3f}' for b in telemetry['beta']]}")
            print("-" * 75)

if __name__ == "__main__":
    print("=== FLUXVM CONTROLLER SWEEP ===")
    print("Baseline OBSERVE target: D_t peaks at ~141.5, Loss ~0.0737\n")
    
    # Sweep k values
    k_vals = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5]
    
    print("--- Sweeping k (contraction strength) with fixed alpha=0.1 ---")
    for k in k_vals:
        run_sweep_step(k=k, alpha=0.1)
        
    print("\n--- Sweeping alpha (memory accumulation speed) with fixed k=0.05 ---")
    alpha_vals = [0.01, 0.05, 0.1, 0.2, 0.5]
    for alpha in alpha_vals:
        run_sweep_step(k=0.05, alpha=alpha)
