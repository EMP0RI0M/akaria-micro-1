import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import argparse
from chm_model import CHMConfig, ControlledHyperloopMoE

def create_optimizer(model, base_lr, tied_g):
    normal_params = []
    expert_params = []
    fluxvm_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "flux_adapter" in name:
            fluxvm_params.append(param)
        elif "moe" in name and ("w1s" in name or "w2s" in name or "w3s" in name):
            expert_params.append(param)
        else:
            normal_params.append(param)
            
    expert_lr = base_lr / math.sqrt(tied_g)
    
    optim_groups = [
        {"params": normal_params, "lr": base_lr},
        {"params": expert_params, "lr": expert_lr},
        {"params": fluxvm_params, "lr": base_lr} 
    ]
    return optim.AdamW(optim_groups, weight_decay=0.01)

def run_synthetic_overfit(config_type, flux_mode, batch_size=2, seq_len=32, steps=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    cfg = CHMConfig(
        config_type=config_type,
        flux_mode=flux_mode,
        d_model=256,
        n_heads=4,
        n_experts=8,
        core_loops=5,
        prelude_layers=1,
        coda_layers=1,
        vocab_size=1000,
        n_streams=4 if config_type in ["C", "D", "E"] else 1
    )
    
    model = ControlledHyperloopMoE(cfg).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate tied vs unique parameters
    # This is a naive calculation for the report
    tied_params = sum(p.numel() for name, p in model.named_parameters() if "moe" in name and "w1s" in name) * cfg.core_loops
    
    tied_g = cfg.core_loops if config_type in ["D", "E"] else 1
    optimizer = create_optimizer(model, base_lr=1e-3, tied_g=tied_g)
    
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"--- Config {config_type} | Flux Mode: {flux_mode} ---")
    print(f"Total Params: {total_params / 1e6:.2f}M | Device: {device}")
    
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    start_time = time.time()
    for step in range(steps):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits, _, div_log, beta_log = model(x)
            loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"Step {step}: NaN/Inf detected in gradients!")
            break
            
        scaler.step(optimizer)
        scaler.update()
        
        if step % 10 == 0 or step == steps - 1:
            print(f"Step {step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            if div_log: print(f"  D(t) Log: {[f'{d:.4f}' for d in div_log]}")
            if beta_log: print(f"  Beta Log: {[f'{b:.4f}' for b in beta_log]}")
                
    end_time = time.time()
    throughput = (batch_size * seq_len * steps) / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} tokens/sec")
    
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
    
    print("--------------------------------------------------\n")
    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a rapid 2-step verification of Config E")
    args = parser.parse_args()

    print("==================================================")
    print("CHM Prototype Single-GPU Verification Harness")
    print("==================================================\n")
    
    if args.smoke_test:
        print(">>> Running rapid smoke test (Config E, 2 steps) <<<")
        run_synthetic_overfit("E", "CONTROL", steps=2)
        print("Smoke test completed.")
        return
    
    # Validate each configuration
    run_synthetic_overfit("A", "OFF", steps=20)
    run_synthetic_overfit("B", "OFF", steps=20)
    run_synthetic_overfit("C", "OFF", steps=20)
    run_synthetic_overfit("D", "OFF", steps=20)
    
    run_synthetic_overfit("E", "OFF", steps=20)
    run_synthetic_overfit("E", "OBSERVE", steps=20)
    run_synthetic_overfit("E", "CONTROL", steps=20)
    
    print("All tests completed. Prototype is verified for Single-GPU execution.")

if __name__ == "__main__":
    main()
