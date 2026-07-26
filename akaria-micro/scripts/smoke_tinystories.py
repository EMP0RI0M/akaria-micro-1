import torch
import torch.nn as nn
import os
import sys
import time
import math
from tokenizers import Tokenizer
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE
from chm.baseline import StandardTransformer

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def get_trainable_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Tokenizer Setup (Smoke Test sizes)
    print("Loading tokenizer (make sure you trained it first!)...")
    if not os.path.exists("tinystories_tokenizer.json"):
        print("ERROR: tinystories_tokenizer.json not found. Run scripts/train_tokenizer.py first.")
        return
        
    tokenizer = Tokenizer.from_file("tinystories_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with vocab size: {vocab_size}")
    
    batch_size = 4
    seq_len = 128
    
    print("Generating synthetic pilot batch (to avoid long dataset download for smoke test)...")
    # For smoke test only: we just need a batch of tokens. The actual train loop will load datasets.
    x_train = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    y_train = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    x_val = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    y_val = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # 2. Define Contestants
    # CHM Config E (matches earlier tests)
    chm_cfg = CHMConfig(
        vocab_size=vocab_size,
        d_model=384,
        n_heads=6,
        prelude_layers=1,
        core_loops=5,
        coda_layers=1,
        n_streams=4,
        n_experts=8,
        experts_per_token=2,
        d_ff=1024,
        expert_hidden_dim=1024,
        sequence_length=seq_len,
        config_type="E"
    )
    
    # 17-layer standard transformer gives ~33.22M parameters (vs 33.80M for CHM)
    baseline_model = StandardTransformer(chm_cfg, num_layers=17)
    
    contestants = [
        ("Baseline", baseline_model),
        ("E0", ControlledHyperloopMoE(CHMConfig(**{**chm_cfg.__dict__, "flux_mode": "E0"}))),
        ("E3", ControlledHyperloopMoE(CHMConfig(**{**chm_cfg.__dict__, "flux_mode": "E3"}))),
        ("E4_0.1", ControlledHyperloopMoE(CHMConfig(**{**chm_cfg.__dict__, "flux_mode": "E4", "lambda_barrier": 0.1}))),
        ("E5", ControlledHyperloopMoE(CHMConfig(**{**chm_cfg.__dict__, "flux_mode": "E5"}))),
    ]
    
    # 3. Report Parameters
    print("\n" + "="*80)
    print("CONTESTANT PARAMETERS")
    print("="*80)
    for name, model in contestants:
        print(f"{name:10s} | Total: {get_parameter_count(model)/1e6:.2f}M | Trainable: {get_trainable_parameter_count(model)/1e6:.2f}M")
        
    # 4. Smoke Test Execution
    print("\n" + "="*80)
    print("RUNNING SMOKE TESTS")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    
    for name, model in contestants:
        print(f"\nTesting {name}...")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # --- TRAIN MODE ---
        model.train()
        start_t = time.time()
        
        # Hidden state hook
        hidden_rms = []
        def hook(module, inp, out):
            hidden_rms.append(out.norm(dim=-1).mean().item())
        h_hook = model.norm_out.register_forward_hook(hook)
        
        # Forward
        logits, telemetry = model(x_train)
        assert not torch.isnan(logits).any(), f"{name} produced NaN in logits!"
        h_hook.remove()
        
        # Logit stats check (before backward)
        with torch.no_grad():
            print(f"  [Stats] Logits Mean: {logits.mean().item():.4f} | Std: {logits.std().item():.4f} | Min: {logits.min().item():.4f} | Max: {logits.max().item():.4f}")
            print(f"  [Stats] Hidden RMS:  {hidden_rms[0]:.4f}")
            
        # Loss
        lm_loss = criterion(logits.view(-1, vocab_size), y_train.view(-1))
        loss = lm_loss
        if "L_barrier" in telemetry and len(telemetry["L_barrier"]) > 0:
            loss = loss + torch.stack(telemetry["L_barrier"]).sum()
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # --- BENCHMARK (Warmup + Run) ---
        if torch.cuda.is_available():
            for _ in range(3):
                _, _ = model(x_train)
            torch.cuda.synchronize()
            start_t = time.time()
            for _ in range(5):
                _, _ = model(x_train)
            torch.cuda.synchronize()
            elapsed = time.time() - start_t
            tps = (5 * batch_size * seq_len) / elapsed
        else:
            tps = 0
        
        print(f"  [Train] Loss: {lm_loss.item():.4f} | Grad: {grad_norm.item():.4f} | Throughput: {tps:.0f} tok/s")
        
        # --- TELEMETRY CHECK ---
        if name != "Baseline":
            assert "D_t" in telemetry, f"Telemetry missing for {name}"
            print(f"  [Tele] D_t len: {len(telemetry['D_t'])}, G_t len: {len(telemetry.get('G_t', []))}")
        
        # --- EVAL MODE ---
        model.eval()
        with torch.no_grad():
            logits_val, _ = model(x_val)
            val_loss = criterion(logits_val.view(-1, vocab_size), y_val.view(-1))
            ppl = math.exp(val_loss.item())
            print(f"  [Eval]  Loss: {val_loss.item():.4f} | PPL: {ppl:.2f}")
            
            # GENERATION SMOKE TEST (Just 1 step to ensure it doesn't crash)
            last_token = x_val[:, -1:]
            gen_logits, _ = model(last_token)
            next_token = torch.argmax(gen_logits[:, -1, :], dim=-1)
            assert next_token.shape == (batch_size,), "Generation shape mismatch"
            print("  [Gen]   1-step generation successful.")
            
        # --- CHECKPOINT SMOKE TEST ---
        ckpt_path = f"smoke_ckpt_{name}.pt"
        torch.save(model.state_dict(), ckpt_path)
        
        model_clone = type(model)(model.cfg if hasattr(model, 'cfg') else chm_cfg, num_layers=17) if name == "Baseline" else type(model)(model.cfg)
        model_clone.load_state_dict(torch.load(ckpt_path, weights_only=True))
        os.remove(ckpt_path)
        print("  [Ckpt]  Save and restore successful.")
        
        # Cleanup memory
        del model, optimizer, logits, loss, telemetry, model_clone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nSmoke tests completed successfully!")

if __name__ == "__main__":
    main()
