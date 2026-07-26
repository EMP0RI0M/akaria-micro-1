import torch
import torch.nn as nn
import os
import sys
import time
import math
import json
import csv
import argparse
from tokenizers import Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE
from chm.baseline import StandardTransformer

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def safe_telemetry_to_tensor(values, device=None):
    """
    Safely converts a list of mixed floats, ints, and tensors into a 1D tensor.
    Does NOT detach or convert in-place if they are tensors (to preserve gradients where needed).
    Instead, creates a separate diagnostic tensor for NaN/Inf checking.
    """
    if not isinstance(values, list):
        values = [values]
        
    extracted = [v.detach().item() if torch.is_tensor(v) else float(v) for v in values]
    return torch.as_tensor(extracted, dtype=torch.float32, device=device)

def check_nan(tensor, name, step, contestant):
    if tensor is not None and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
        raise RuntimeError(f"NaN/Inf detected in {name} for contestant {contestant} at step {step}!")

def generate_samples(model, tokenizer, device, prompts=["Once upon a time", "A little girl named"], max_new_tokens=30):
    model.eval()
    results = []
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt).ids
        x = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(x)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
                x = torch.cat([x, next_token], dim=1)
                
        output_text = tokenizer.decode(x[0].tolist())
        results.append(output_text)
    return results

def train_contestant(name, model_fn, tokenizer, train_loader, val_loader, device, total_steps, eval_interval):
    print(f"\n{'='*80}\nStarting Training for {name}\n{'='*80}")
    set_seed(42) # Ensure fair initialization
    model = model_fn().to(device)
    vocab_size = tokenizer.get_vocab_size()
    
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0
        
    print(f"Parameters: {get_parameter_count(model)/1e6:.2f}M | PAD Token: {pad_id}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    log_file = f"metrics_{name}.csv"
    
    global_step = 0
    tokens_processed = 0
    
    latest_ckpt_path = f"ckpt_latest_{name}.pt"
    if os.path.exists(latest_ckpt_path):
        print(f"[{name}] Found existing checkpoint {latest_ckpt_path}. Resuming...")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["step"]
        print(f"[{name}] Resumed from step {global_step}.")
        # We append to the log file instead of overwriting
        log_mode = "a"
    else:
        log_mode = "w"

    with open(log_file, log_mode, newline="") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(["Step", "Train_Loss", "Val_Loss", "Val_PPL", "Tok_Sec", "D_t", "G_t_Act", "I_t_Sat"])
        
    train_iter = iter(train_loader)
    
    while global_step < total_steps:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        x_train = batch["input_ids"][:, :-1].to(device)
        y_raw = batch["input_ids"][:, 1:].to(device)
        
        # Apply padding mask: set PAD targets to -100 so CrossEntropy ignores them
        y_train = torch.where(y_raw == pad_id, torch.tensor(-100, device=device), y_raw)
        
        optimizer.zero_grad()
        step_start = time.time()
        logits, telemetry = model(x_train)
        
        check_nan(logits, "logits", global_step, name)
        
        lm_loss = criterion(logits.reshape(-1, vocab_size), y_train.reshape(-1))
        check_nan(lm_loss, "lm_loss", global_step, name)
        loss = lm_loss
        
        if "L_barrier" in telemetry and len(telemetry["L_barrier"]) > 0:
            barrier_list = telemetry["L_barrier"]
            # Diagnostic check using detached values
            check_nan(safe_telemetry_to_tensor(barrier_list), "L_barrier", global_step, name)
            
            # Loss accumulation preserves graph if tensors are differentiable
            barrier_loss = 0
            for b in barrier_list:
                barrier_loss = barrier_loss + b
            loss = loss + barrier_loss
            
        loss.backward()
        
        # Check gradients
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                check_nan(param.grad, f"grad_{param_name}", global_step, name)
                
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Count non-padded tokens
        valid_tokens = (y_train != -100).sum().item()
        tokens_processed += valid_tokens
        tps = valid_tokens / max((time.time() - step_start), 1e-6)
        
        if global_step % 50 == 0 and global_step != 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[{name}] Step {global_step}/{total_steps} | LM Loss: {lm_loss.item():.4f} | LR: {current_lr:.2e} | Tok/s: {tps:.0f}")
        
        if global_step % eval_interval == 0 or global_step == total_steps - 1:
            # Eval
            model.eval()
            val_loss_sum = 0
            val_batches = 0
            with torch.no_grad():
                for i, v_batch in enumerate(val_loader):
                    if i >= 10: # Limit eval size for speed
                        break
                    x_val = v_batch["input_ids"][:, :-1].to(device)
                    v_raw = v_batch["input_ids"][:, 1:].to(device)
                    y_val = torch.where(v_raw == pad_id, torch.tensor(-100, device=device), v_raw)
                    
                    v_logits, _ = model(x_val)
                    v_loss = criterion(v_logits.reshape(-1, vocab_size), y_val.reshape(-1))
                    val_loss_sum += v_loss.item()
                    val_batches += 1
            
            val_loss = val_loss_sum / max(val_batches, 1)
            val_ppl = math.exp(min(val_loss, 20)) # Cap at e^20 for reporting
            
            # Extract basic telemetry stats
            d_mean = 0.0
            g_act = 0.0
            i_sat = 0.0
            if "D_t" in telemetry and len(telemetry["D_t"]) > 0:
                dt_tensor = safe_telemetry_to_tensor(telemetry["D_t"])
                check_nan(dt_tensor, "D_t", global_step, name)
                d_mean = dt_tensor.mean().item()
            if "G_t" in telemetry and len(telemetry["G_t"]) > 0:
                gt_tensor = safe_telemetry_to_tensor(telemetry["G_t"])
                check_nan(gt_tensor, "G_t", global_step, name)
                g_act = (gt_tensor > 0).float().mean().item()
            if "I_t" in telemetry and len(telemetry["I_t"]) > 0:
                it_tensor = safe_telemetry_to_tensor(telemetry["I_t"])
                check_nan(it_tensor, "I_t", global_step, name)
                i_sat = (it_tensor >= 0.999).float().mean().item()
                
            print(f"[{name}] EVAL Step {global_step} | Val CE: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | D_t: {d_mean:.2f} | Sat: {i_sat:.2f}")
            
            # Generate
            samples = generate_samples(model, tokenizer, device)
            print(f"  Sample: {samples[0]}")
            
            # Log
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([global_step, lm_loss.item(), val_loss, val_ppl, tps, d_mean, g_act, i_sat])
                
            # Checkpoint (Save full training state)
            ckpt = {
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt, f"ckpt_{name}_step{global_step}.pt")
            torch.save(ckpt, latest_ckpt_path)
            
        elif global_step % 250 == 0 and global_step != 0:
            # Save intermediate resumable checkpoint without full eval
            ckpt = {
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt, latest_ckpt_path)
            
        global_step += 1
        
    del model, optimizer, scheduler, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run 50 step short pilot mode")
    args = parser.parse_args()
    
    total_steps = 50 if args.pilot else 5000
    eval_interval = 10 if args.pilot else 500
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Pilot Mode: {args.pilot} | Total Steps: {total_steps} | Eval Interval: {eval_interval}")
    
    set_seed(42)
    
    tokenizer = Tokenizer.from_file("tinystories_tokenizer.json")
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0
        
    def tokenize_function(examples):
        outputs = [tokenizer.encode(text).ids for text in examples["text"]]
        seq_len = 256
        padded = []
        for out in outputs:
            if len(out) >= seq_len + 1:
                padded.append(out[:seq_len + 1])
            else:
                padded.append(out + [pad_id] * (seq_len + 1 - len(out)))
        return {"input_ids": padded}

    print("Preparing dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    train_ds = dataset["train"].select(range(50000)).map(tokenize_function, batched=True, remove_columns=["text"])
    train_ds.set_format(type="torch", columns=["input_ids"])
    # MUST set worker_init_fn and generator for reproducible dataloader
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g)
    
    val_ds = dataset["validation"].select(range(1000)).map(tokenize_function, batched=True, remove_columns=["text"])
    val_ds.set_format(type="torch", columns=["input_ids"])
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    chm_cfg = CHMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=384, n_heads=6, prelude_layers=1, core_loops=5, coda_layers=1,
        n_streams=4, n_experts=8, experts_per_token=2, d_ff=1024, expert_hidden_dim=1024,
        sequence_length=256, config_type="E"
    )
    
    contestants = [
        ("Baseline", lambda: StandardTransformer(chm_cfg, num_layers=17)),
        ("E0", lambda: ControlledHyperloopMoE(CHMConfig(**{**chm_cfg.__dict__, "flux_mode": "E0"}))),
        ("E3", lambda: ControlledHyperloopMoE(CHMConfig(**{**chm_cfg.__dict__, "flux_mode": "E3"}))),
        ("E4_0.1", lambda: ControlledHyperloopMoE(CHMConfig(**{**chm_cfg.__dict__, "flux_mode": "E4", "lambda_barrier": 0.1}))),
        ("E5", lambda: ControlledHyperloopMoE(CHMConfig(**{**chm_cfg.__dict__, "flux_mode": "E5"}))),
    ]
    
    for name, model_fn in contestants:
        train_contestant(name, model_fn, tokenizer, train_loader, val_loader, device, total_steps, eval_interval)
        
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
