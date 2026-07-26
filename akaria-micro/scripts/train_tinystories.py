import torch
import torch.nn as nn
import os
import sys
import time
import math
import json
import csv
from tokenizers import Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE
from chm.baseline import StandardTransformer

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def generate_samples(model, tokenizer, device, prompts=["Once upon a time", "A little girl named"], max_new_tokens=30):
    model.eval()
    results = []
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

def train_contestant(name, model_fn, tokenizer, train_loader, val_loader, device, total_steps=5000, eval_interval=500):
    print(f"\n{'='*80}\nStarting Training for {name}\n{'='*80}")
    model = model_fn().to(device)
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"Parameters: {get_parameter_count(model)/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    log_file = f"metrics_{name}.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Train_Loss", "Val_Loss", "Val_PPL", "Tok_Sec", "D_t", "G_t_Act", "I_t_Sat"])
        
    global_step = 0
    tokens_processed = 0
    start_time = time.time()
    
    train_iter = iter(train_loader)
    
    while global_step < total_steps:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        x_train = batch["input_ids"][:, :-1].to(device)
        y_train = batch["input_ids"][:, 1:].to(device)
        
        optimizer.zero_grad()
        step_start = time.time()
        logits, telemetry = model(x_train)
        
        lm_loss = criterion(logits.reshape(-1, vocab_size), y_train.reshape(-1))
        loss = lm_loss
        if "L_barrier" in telemetry and len(telemetry["L_barrier"]) > 0:
            loss = loss + torch.stack(telemetry["L_barrier"]).sum()
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        tokens_processed += x_train.numel()
        tps = x_train.numel() / (time.time() - step_start)
        
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
                    y_val = v_batch["input_ids"][:, 1:].to(device)
                    v_logits, _ = model(x_val)
                    v_loss = criterion(v_logits.reshape(-1, vocab_size), y_val.reshape(-1))
                    val_loss_sum += v_loss.item()
                    val_batches += 1
            
            val_loss = val_loss_sum / val_batches
            val_ppl = math.exp(min(val_loss, 20)) # Cap at e^20 for reporting
            
            # Extract basic telemetry stats
            d_mean = 0.0
            g_act = 0.0
            i_sat = 0.0
            if "D_t" in telemetry and len(telemetry["D_t"]) > 0:
                d_mean = sum(telemetry["D_t"]) / len(telemetry["D_t"])
            if "G_t" in telemetry and len(telemetry["G_t"]) > 0:
                g_act = sum(1 for g in telemetry["G_t"] if g > 0) / len(telemetry["G_t"])
            if "I_t" in telemetry and len(telemetry["I_t"]) > 0:
                i_sat = sum(1 for i in telemetry["I_t"] if i >= 0.999) / len(telemetry["I_t"])
                
            print(f"Step {global_step:5d} | Train: {lm_loss.item():.4f} | Val CE: {val_loss:.4f} | Val PPL: {val_ppl:6.2f} | Tok/s: {tps:5.0f} | D_t: {d_mean:.2f} | Sat: {i_sat:.2f}")
            
            # Generate
            samples = generate_samples(model, tokenizer, device)
            print(f"  Sample: {samples[0]}")
            
            # Log
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([global_step, lm_loss.item(), val_loss, val_ppl, tps, d_mean, g_act, i_sat])
                
            # Checkpoint
            torch.save(model.state_dict(), f"ckpt_{name}_step{global_step}.pt")
            
        global_step += 1
        
    del model, optimizer, scheduler, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = Tokenizer.from_file("tinystories_tokenizer.json")
    
    def tokenize_function(examples):
        outputs = [tokenizer.encode(text).ids for text in examples["text"]]
        # Pad or truncate to seq_len + 1 (for shifted targets)
        seq_len = 256
        padded = []
        for out in outputs:
            if len(out) >= seq_len + 1:
                padded.append(out[:seq_len + 1])
            else:
                padded.append(out + [0] * (seq_len + 1 - len(out)))
        return {"input_ids": padded}

    print("Preparing dataset...")
    # Loading a subset for rapid experimentation, in a real run use the full dataset
    dataset = load_dataset("roneneldan/TinyStories")
    
    train_ds = dataset["train"].select(range(50000)).map(tokenize_function, batched=True, remove_columns=["text"])
    train_ds.set_format(type="torch", columns=["input_ids"])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
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
        train_contestant(name, model_fn, tokenizer, train_loader, val_loader, device, total_steps=5000, eval_interval=1000)

if __name__ == "__main__":
    main()
