import torch
import torch.nn as nn
import time
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE

def run_multiseed_ablation(configs, seeds=[42, 43, 44, 45, 46], batch_size=2, seq_len=32, steps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Multiseed Sweep ===")
    
    results = {cfg_name: [] for cfg_name, _ in configs}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for cfg_name, l_bar in configs:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Same synthetic batches across configs
            x = torch.randint(0, 100, (batch_size, seq_len)).to(device)
            y = torch.randint(0, 100, (batch_size, seq_len)).to(device)
            
            cfg = CHMConfig(config_type="E", flux_mode=cfg_name, lambda_barrier=l_bar, vocab_size=100)
            
            torch.manual_seed(seed) # Ensure identical model init
            model = ControlledHyperloopMoE(cfg).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            model.train()
            
            # Metrics to track
            lm_loss_final = 0.0
            grad_norm_final = 0.0
            D_t_list = []
            active_steps = 0
            sat_steps = 0
            total_steps = 0
            barrier_pass_list = []
            
            start_time = time.time()
            for step in range(steps):
                optimizer.zero_grad()
                logits, telemetry = model(x)
                lm_loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
                
                loss = lm_loss
                if "L_barrier" in telemetry and telemetry["L_barrier"]:
                    L_b = torch.stack(telemetry["L_barrier"]).sum()
                    loss = loss + L_b
                    
                loss.backward()
                
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if step == steps - 1:
                    lm_loss_final = lm_loss.item()
                    grad_norm_final = grad_norm.item()
                    
                # Collect telemetry stats across all recurrent steps
                if "D_t" in telemetry:
                    for d in telemetry["D_t"]:
                        D_t_list.append(d.item())
                if "G_t" in telemetry:
                    for g in telemetry["G_t"]:
                        if g.item() > 0:
                            active_steps += 1
                        total_steps += 1
                if "I_t" in telemetry:
                    for i_val in telemetry["I_t"]:
                        if i_val.item() >= 0.999:
                            sat_steps += 1
                if "barrier_pass" in telemetry:
                    for bp in telemetry["barrier_pass"]:
                        # Skip NaNs from E0
                        if not torch.isnan(bp):
                            barrier_pass_list.append(bp.item())
                            
            throughput = (batch_size * seq_len * steps) / (time.time() - start_time)
            
            # Final stats for this run
            mean_D = np.mean(D_t_list) if D_t_list else 0.0
            frac_active = active_steps / total_steps if total_steps > 0 else 0.0
            frac_sat = sat_steps / total_steps if total_steps > 0 else 0.0
            mean_bp = np.mean(barrier_pass_list) if barrier_pass_list else float('nan')
            
            results[cfg_name].append({
                "loss": lm_loss_final,
                "grad": grad_norm_final,
                "mean_D": mean_D,
                "frac_active": frac_active,
                "frac_sat": frac_sat,
                "mean_bp": mean_bp,
                "throughput": throughput
            })
            print(f"{cfg_name:2s} | Loss: {lm_loss_final:.4f} | Grad: {grad_norm_final:6.2f} | Act: {frac_active:.2f} | Sat: {frac_sat:.2f}")

    print("\n" + "="*80)
    print("FINAL MULTI-SEED AGGREGATES (Mean ± Std)")
    print(f"{'Mode':<5} | {'LM Loss':<15} | {'Grad Norm':<15} | {'D_t':<10} | {'% Active':<10} | {'% Sat':<10} | {'BP Rate':<10} | {'Tok/s'}")
    print("-" * 80)
    
    for cfg_name, _ in configs:
        runs = results[cfg_name]
        loss = [r["loss"] for r in runs]
        grad = [r["grad"] for r in runs]
        D = [r["mean_D"] for r in runs]
        act = [r["frac_active"] for r in runs]
        sat = [r["frac_sat"] for r in runs]
        bp = [r["mean_bp"] for r in runs]
        tps = [r["throughput"] for r in runs]
        
        # Format BP rate correctly (handle NaNs for E0)
        bp_valid = [x for x in bp if not np.isnan(x)]
        if bp_valid:
            bp_str = f"{np.mean(bp_valid):.2f}±{np.std(bp_valid):.2f}"
        else:
            bp_str = "N/A"
            
        print(f"{cfg_name:<5} | {np.mean(loss):.4f}±{np.std(loss):.4f} | {np.mean(grad):6.2f}±{np.std(grad):.2f} | {np.mean(D):.2f}±{np.std(D):.2f} | "
              f"{np.mean(act):.2f}±{np.std(act):.2f} | {np.mean(sat):.2f}±{np.std(sat):.2f} | {bp_str:<10} | {np.mean(tps):.0f}")

if __name__ == "__main__":
    configs = [
        ("E0", 0.0), # Observe
        ("E2", 0.0), # P/D threshold
        ("E3", 0.0), # PID + Memory
        ("E4", 0.0), # PID + Memory + Microsteps
        ("E4", 0.1), # Same, with lambda=0.1
        ("E5", 0.0), # Triggered hard-reset ablation
    ]
    run_multiseed_ablation(configs)
