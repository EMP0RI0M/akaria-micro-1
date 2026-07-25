import torch
import torch.nn as nn
import time
from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE

def run_synthetic_ablation(cfg: CHMConfig, batch_size=2, seq_len=32, steps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    model = ControlledHyperloopMoE(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Tiny synthetic overfit
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    
    start_time = time.time()
    for step in range(steps):
        optimizer.zero_grad()
        logits, telemetry = model(x)
        loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step == steps - 1:
            print(f"--- Config {cfg.config_type} | FluxMode: {cfg.flux_mode} ---")
            print(f"Total Params: {total_params / 1e6:.2f}M")
            print(f"Final Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            if telemetry["D_stream"]: print(f"D_stream(t): {[f'{d:.4f}' for d in telemetry['D_stream']]}")
            if telemetry["beta"]: print(f"Beta(t): {[f'{b:.4f}' for b in telemetry['beta']]}")
            
    throughput = (batch_size * seq_len * steps) / (time.time() - start_time)
    print(f"Throughput: {throughput:.2f} tok/s")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
    print("-" * 50)

if __name__ == "__main__":
    configs = [
        ("A", "OFF"), ("B", "OFF"), ("C", "OFF"), ("D", "OFF"),
        ("E", "OFF"), ("E", "OBSERVE"), ("E", "CONTROL")
    ]
    for c_type, flux in configs:
        cfg = CHMConfig(config_type=c_type, flux_mode=flux)
        run_synthetic_ablation(cfg)
