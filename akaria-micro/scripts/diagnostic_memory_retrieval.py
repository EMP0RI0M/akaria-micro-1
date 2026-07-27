import torch
import torch.nn as nn
import time
import csv
import os
from chm.config import CHMConfig
from chm.memory import E0Memory

def get_vram_usage():
    if not torch.cuda.is_available():
        return 0, 0
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return allocated, reserved

def generate_diagnostic_batch(batch_size, mode, device='cpu'):
    """
    mode 'A': logical_seq_len=256, chunk_size=256. 1 chunk. Multiple distractors.
    mode 'B': logical_seq_len=512, chunk_size=256. 2 chunks. ONE pair in chunk 1, query in chunk 2.
    mode 'C': logical_seq_len=512, chunk_size=256. 2 chunks. Multiple pairs in chunk 1, query in chunk 2.
    """
    if mode == 'A':
        logical_seq_len = 256
        num_pairs = 4
    elif mode == 'B':
        logical_seq_len = 512
        num_pairs = 1
    elif mode == 'C':
        logical_seq_len = 512
        num_pairs = 4
    else:
        raise ValueError("Invalid mode")

    chunk_size = 256
    inputs = torch.randint(3001, 16000, (batch_size, logical_seq_len), device=device)
    targets = torch.full((batch_size, logical_seq_len), -100, dtype=torch.long, device=device)
    
    query_marker = 999
    
    info = []
    
    for b in range(batch_size):
        keys = torch.randperm(2000)[:num_pairs] + 1000
        values = torch.randperm(2000)[:num_pairs] + 1000
        
        if mode == 'A':
            # Place pairs randomly in the first half of chunk 1
            positions = torch.randperm(100)[:num_pairs]
            for i in range(num_pairs):
                pos = positions[i].item()
                inputs[b, pos] = keys[i]
                inputs[b, pos+1] = values[i]
            
            target_idx = torch.randint(0, num_pairs, (1,)).item()
            target_key = keys[target_idx]
            target_val = values[target_idx]
            
            # Place query near the end of chunk 1
            query_pos = 256 - 3
            
        else: # B or C
            # Place pairs randomly in chunk 1
            positions = torch.randperm(250)[:num_pairs]
            for i in range(num_pairs):
                pos = positions[i].item()
                inputs[b, pos] = keys[i]
                inputs[b, pos+1] = values[i]
                
            target_idx = torch.randint(0, num_pairs, (1,)).item()
            target_key = keys[target_idx]
            target_val = values[target_idx]
            
            # Place query near the end of chunk 2
            query_pos = 512 - 3
            
        inputs[b, query_pos] = query_marker
        inputs[b, query_pos + 1] = target_key
        targets[b, query_pos + 2] = target_val
        
        info.append({
            'key': target_key.item(),
            'val': target_val.item(),
            'query_pos': query_pos
        })
        
    return inputs, targets, info

def run_experiment(mode, steps=500, batch_size=8):
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT CONTROL {mode}")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = CHMConfig(
        vocab_size=16384,
        d_model=384,
        n_heads=6,
        prelude_layers=1,
        core_loops=5,
        coda_layers=1,
        config_type="E",
        sequence_length=512
    )
    
    model = E0Memory(cfg, num_mem_tokens=16, mem_refinement_steps=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    logical_seq_len = 256 if mode == 'A' else 512
    chunk_size = 256
    
    # Random chance for a uniform choice over 2000 possible values
    random_chance = (1.0 / 2000.0) * 100
    
    initial_loss = None
    final_loss = None
    final_train_acc = 0.0
    
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        
        inputs, targets, batch_info = generate_diagnostic_batch(batch_size, mode, device=device)
        
        # Track initial memory state
        mem_before = model.get_initial_memory(batch_size, device).detach()
        
        # We can just process it in one go through the forward wrapper
        logits, telemetry = model(inputs, chunk_size=chunk_size, detach_memory_every=0, return_last_chunk_only=True)
        
        # Memory tracking
        if mode in ['B', 'C']:
            # The memory state after chunk 1 is what's passed to chunk 2
            mem_after_chunk1 = model.forward_chunk(inputs[:, :256], mem_before)[2]
            
            # The write gate value during chunk 1
            g_t = torch.sigmoid(model.w_g(mem_after_chunk1)) # approximate
            gate_mean = g_t.mean().item()
            gate_std = g_t.std().item()
            
            # Cosine similarity
            cos_sim = nn.functional.cosine_similarity(mem_before.view(batch_size, -1), mem_after_chunk1.view(batch_size, -1), dim=1).mean().item()
            mem_state_mean = mem_after_chunk1.mean().item()
            mem_state_std = mem_after_chunk1.std().item()
            
        targets_last_chunk = targets[:, -chunk_size:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets_last_chunk[:, 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
        
        if step == 1:
            initial_loss = loss.item()
            
        loss.backward()
        
        # Diagnostics: gradients
        grad_norm_mem_params = 0.0
        for name, p in model.named_parameters():
            if 'w_g' in name or 'g_theta' in name or 'M_0' in name:
                if p.grad is not None:
                    grad_norm_mem_params += p.grad.norm().item() ** 2
        grad_norm_mem_params = grad_norm_mem_params ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            preds = torch.argmax(shift_logits, dim=-1)
            mask = shift_labels != -100
            correct = (preds[mask] == shift_labels[mask]).sum().item()
            total = mask.sum().item()
            train_acc = correct / max(total, 1) * 100
            
        if step % 100 == 0 or step == 1:
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Acc: {train_acc:5.1f}% | MemGrad: {grad_norm_mem_params:.4f}")
            if step == steps:
                final_train_acc = train_acc
                final_loss = loss.item()
                
                print("\nExamples from final step:")
                for b in range(min(3, batch_size)):
                    info = batch_info[b]
                    q_pos_in_last_chunk = info['query_pos'] % 256
                    pred_token = preds[b, q_pos_in_last_chunk].item()
                    actual_target = shift_labels[b, q_pos_in_last_chunk].item()
                    print(f"  Example {b}: KEY={info['key']}, VALUE={info['val']}, QUERY={info['key']}")
                    print(f"             TARGET={actual_target}, MODEL PREDICTION={pred_token}")
                    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for _ in range(10):
            v_in, v_tgt, _ = generate_diagnostic_batch(batch_size, mode, device=device)
            v_log, _ = model(v_in, chunk_size=chunk_size, detach_memory_every=0, return_last_chunk_only=True)
            s_log = v_log[:, :-1, :].contiguous()
            s_lbl = v_tgt[:, -chunk_size:][:, 1:].contiguous()
            p = torch.argmax(s_log, dim=-1)
            m = s_lbl != -100
            val_correct += (p[m] == s_lbl[m]).sum().item()
            val_total += m.sum().item()
    val_acc = val_correct / max(val_total, 1) * 100
    
    alloc, res = get_vram_usage()
    param_count = sum(p.numel() for p in model.parameters())
    
    print("\n" + "-"*60)
    print(f"CONTROL {mode} DIAGNOSTICS")
    print("-"*60)
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Train Accuracy: {final_train_acc:.1f}%")
    print(f"Validation Accuracy: {val_acc:.1f}%")
    print(f"Random Chance Accuracy: {random_chance:.4f}%")
    if mode in ['B', 'C']:
        print(f"Mem Grad Norm: {grad_norm_mem_params:.6f}")
        print(f"Mem State Mean: {mem_state_mean:.6f}, Std: {mem_state_std:.6f}")
        print(f"Gate Mean: {gate_mean:.6f}, Std: {gate_std:.6f}")
        print(f"Cosine Sim (M_before, M_after): {cos_sim:.6f}")
    print(f"VRAM: {alloc:.2f} GB")
    print(f"Parameters: {param_count / 1e6:.2f}M")
    
    return val_acc > 10.0 # Return whether it successfully learned

if __name__ == "__main__":
    success_A = run_experiment('A')
    if not success_A:
        print("\nCONTROL A FAILED. Stopping experiments. Task/benchmark formulation is broken.")
    else:
        success_B = run_experiment('B')
        if success_B:
            run_experiment('C')
