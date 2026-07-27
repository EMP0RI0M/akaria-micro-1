import torch
import torch.nn as nn
import time
import csv
import os
from chm.config import CHMConfig
from chm.memory import E0Memory

def generate_retrieval_batch(batch_size, logical_seq_len, chunk_size=256, num_pairs=4, device='cpu'):
    # Fill with random filler
    inputs = torch.randint(3001, 16000, (batch_size, logical_seq_len), device=device)
    targets = torch.full((batch_size, logical_seq_len), -100, dtype=torch.long, device=device)
    
    query_marker = 999
    
    for b in range(batch_size):
        # Keys and values from the exact same distribution
        keys = torch.randperm(2000)[:num_pairs] + 1000
        values = torch.randperm(2000)[:num_pairs] + 1000
        
        # Place pairs randomly in the first chunk
        positions = torch.randperm(chunk_size - 2)[:num_pairs]
        for i in range(num_pairs):
            pos = positions[i].item()
            inputs[b, pos] = keys[i]
            inputs[b, pos+1] = values[i]
            
        # Target pair
        target_idx = torch.randint(0, num_pairs, (1,)).item()
        target_key = keys[target_idx]
        target_val = values[target_idx]
        
        # Place query at the end (must be in the last chunk)
        query_pos = logical_seq_len - 3
        inputs[b, query_pos] = query_marker
        inputs[b, query_pos + 1] = target_key
        
        # Predict the token AFTER the key
        targets[b, query_pos + 2] = target_val
        
    return inputs, targets

def get_vram_usage():
    if not torch.cuda.is_available():
        return 0, 0
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved

def main():
    print("="*60)
    print("E0-MEMORY 512 RETRIEVAL BENCHMARK")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vram_alloc_start, vram_res_start = get_vram_usage()
    print(f"VRAM at beginning: {vram_alloc_start:.2f} GB allocated, {vram_res_start:.2f} GB reserved")
    
    cfg = CHMConfig(
        vocab_size=16384,
        d_model=384,
        n_heads=6,
        prelude_layers=1,
        core_loops=5,
        coda_layers=1,
        config_type="E",
        sequence_length=512 # physical max for RoPE
    )
    
    model = E0Memory(cfg, num_mem_tokens=16, mem_refinement_steps=1).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    batch_size = 8
    logical_seq_len = 512
    chunk_size = 256
    
    log_file = "memory_retrieval_512_metrics.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Train_Loss", "Train_Acc", "Val_Acc", "VRAM_Alloc_GB"])
        
    best_val_acc = -1.0
    initial_loss = None
    final_loss = None
    final_val_acc = 0.0
    
    consecutive_successes = 0
    solved = False
    
    print("-" * 60)
    
    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()
        
        inputs, targets = generate_retrieval_batch(batch_size, logical_seq_len, chunk_size, device=device)
        
        # Full BPTT for 512 (detach_memory_every=0) to ensure gradients flow back to chunk 1
        # Use return_last_chunk_only=True to prevent VRAM accumulation of early chunk logits
        logits_last_chunk, _ = model(inputs, chunk_size=chunk_size, detach_memory_every=0, return_last_chunk_only=True)
        
        # Calculate loss only on the last chunk
        targets_last_chunk = targets[:, -chunk_size:]
        shift_logits = logits_last_chunk[:, :-1, :].contiguous()
        shift_labels = targets_last_chunk[:, 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
        
        if step == 1:
            initial_loss = loss.item()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Calculate train accuracy
        with torch.no_grad():
            preds = torch.argmax(shift_logits, dim=-1)
            mask = shift_labels != -100
            correct = (preds[mask] == shift_labels[mask]).sum().item()
            total = mask.sum().item()
            train_acc = correct / max(total, 1) * 100
            
        if step % 25 == 0 or step == 1:
            alloc, res = get_vram_usage()
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:5.1f}% | VRAM Alloc: {alloc:.2f}GB | Res: {res:.2f}GB")
            
        # Validation
        if step % 50 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for _ in range(10): # 10 val batches
                    v_in, v_tgt = generate_retrieval_batch(batch_size, logical_seq_len, chunk_size, device=device)
                    v_log, _ = model(v_in, chunk_size=chunk_size, detach_memory_every=0, return_last_chunk_only=True)
                    
                    s_log = v_log[:, :-1, :].contiguous()
                    s_lbl = v_tgt[:, -chunk_size:][:, 1:].contiguous()
                    
                    p = torch.argmax(s_log, dim=-1)
                    m = s_lbl != -100
                    val_correct += (p[m] == s_lbl[m]).sum().item()
                    val_total += m.sum().item()
                    
            val_acc = val_correct / max(val_total, 1) * 100
            final_val_acc = val_acc
            print(f" ---> EVAL Step {step}: Validation Acc: {val_acc:.1f}%")
            
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                alloc, _ = get_vram_usage()
                writer.writerow([step, loss.item(), train_acc, val_acc, alloc])
                
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc
            }
            
            # Save latest
            torch.save(ckpt, "memory_retrieval_512_latest.pt.tmp")
            os.replace("memory_retrieval_512_latest.pt.tmp", "memory_retrieval_512_latest.pt")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(ckpt, "memory_retrieval_512_best.pt.tmp")
                os.replace("memory_retrieval_512_best.pt.tmp", "memory_retrieval_512_best.pt")
                
            if val_acc >= 90.0:
                consecutive_successes += 1
                if consecutive_successes >= 3:
                    print("\n[SUCCESS] Validation retrieval accuracy >= 90% for 3 consecutive evaluations.")
                    print("512-token retrieval declared SOLVED!")
                    solved = True
                    final_loss = loss.item()
                    break
            else:
                consecutive_successes = 0
                
        final_loss = loss.item()

    # End reporting
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_vram = 0.0
        
    vram_alloc_end, _ = get_vram_usage()
    
    print("\n" + "="*60)
    print("E0-MEMORY 512 RETRIEVAL RESULT")
    print("="*60)
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.1f}%")
    print(f"VRAM at End: {vram_alloc_end:.2f} GB")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Parameters: {param_count / 1e6:.2f}M")
    print(f"Solved: {'YES' if solved else 'NO'}")
    print("="*60)
    
    if solved:
        print("\nArchitecture passed Stage 1.")
        print("Stopping benchmark as requested. Please verify no data leakage before scaling.")

if __name__ == "__main__":
    main()
