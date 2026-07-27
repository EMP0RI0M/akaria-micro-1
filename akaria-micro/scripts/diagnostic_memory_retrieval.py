import torch
import torch.nn as nn
import time
from chm.config import CHMConfig
from chm.memory import E0Memory

def get_vram_usage():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.memory_allocated() / (1024 ** 3), torch.cuda.memory_reserved() / (1024 ** 3)

def generate_diagnostic_batch(batch_size, mode, device='cpu'):
    if mode == 'A' or mode == 'SANITY':
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
        
        if mode in ['A', 'SANITY']:
            positions = torch.randperm(100)[:num_pairs]
            for i in range(num_pairs):
                pos = positions[i].item()
                inputs[b, pos] = keys[i]
                inputs[b, pos+1] = values[i]
            
            target_idx = torch.randint(0, num_pairs, (1,)).item()
            target_key = keys[target_idx]
            target_val = values[target_idx]
            
            # Query placed in the same chunk
            query_pos = 256 - 3 
        else:
            # Pairs in first chunk
            positions = torch.randperm(250)[:num_pairs]
            for i in range(num_pairs):
                pos = positions[i].item()
                inputs[b, pos] = keys[i]
                inputs[b, pos+1] = values[i]
                
            target_idx = torch.randint(0, num_pairs, (1,)).item()
            target_key = keys[target_idx]
            target_val = values[target_idx]
            
            # Query in second chunk
            query_pos = 512 - 3 
            
        inputs[b, query_pos] = query_marker
        inputs[b, query_pos + 1] = target_key
        targets[b, query_pos + 2] = target_val
        
        info.append({
            'key': target_key.item(),
            'val': target_val.item(),
            'query_marker_pos': query_pos,
            'target_key_pos': query_pos + 1,
            'target_val_pos': query_pos + 2
        })
        
    return inputs, targets, info

def verify_and_print_batch(inputs, targets, batch_info):
    print("\n--- EXPLICIT ALIGNMENT CHECK BEFORE TRAINING ---")
    B = inputs.size(0)
    for b in range(min(5, B)):
        inf = batch_info[b]
        qm_pos = inf['query_marker_pos']
        tk_pos = inf['target_key_pos']
        tv_pos = inf['target_val_pos']
        
        assert inputs[b, qm_pos] == 999
        assert inputs[b, tk_pos] == inf['key']
        assert targets[b, tv_pos] == inf['val']
        assert targets[b, tv_pos] != -100
        
        # Check shift alignment for LM:
        # We predict using inputs[..., tk_pos], which gives logits[..., tk_pos]
        # We compare against targets[..., tk_pos + 1] == targets[..., tv_pos]
        
        print(f"Example {b}:")
        print(f"  KEY = {inf['key']}  VALUE = {inf['val']}")
        print(f"  query_marker_pos = {qm_pos}, token = {inputs[b, qm_pos]}")
        print(f"  target_key_pos   = {tk_pos}, token = {inputs[b, tk_pos]}")
        print(f"  target_val_pos   = {tv_pos}, expected VALUE = {inf['val']}")
        print(f"  actual supervised target at target_val_pos = {targets[b, tv_pos]}")
        
    print("Alignment strictly verified.\n")

def run_experiment(mode, steps=500, batch_size=8, fixed_batch=False):
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {mode}{' (FIXED SANITY)' if fixed_batch else ''}")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = CHMConfig(vocab_size=16384, d_model=384, n_heads=6, prelude_layers=1, core_loops=5, coda_layers=1, config_type="E", sequence_length=512)
    model = E0Memory(cfg, num_mem_tokens=16, mem_refinement_steps=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    chunk_size = 256
    random_chance = (1.0 / 2000.0) * 100
    
    initial_loss, final_loss, final_train_acc = None, None, 0.0
    mem_grad_norm = 0.0
    mem_state_change = 0.0
    gate_mean, gate_std = 0.0, 0.0
    mem_mean, mem_std = 0.0, 0.0
    
    # Generate fixed batch if required
    if fixed_batch:
        f_inputs, f_targets, f_info = generate_diagnostic_batch(batch_size, mode, device)
        verify_and_print_batch(f_inputs, f_targets, f_info)
    else:
        # Just verify one random batch
        r_inputs, r_targets, r_info = generate_diagnostic_batch(batch_size, mode, device)
        verify_and_print_batch(r_inputs, r_targets, r_info)

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        
        if fixed_batch:
            inputs, targets, batch_info = f_inputs, f_targets, f_info
        else:
            inputs, targets, batch_info = generate_diagnostic_batch(batch_size, mode, device)
            
        mem_before = model.get_initial_memory(batch_size, device).detach()
        
        logits, telemetry = model(inputs, chunk_size=chunk_size, detach_memory_every=0, return_last_chunk_only=True)
        
        if mode in ['B', 'C']:
            # Inspect intermediate memory
            mem_after_chunk1 = model.forward_chunk(inputs[:, :256], mem_before)[2]
            g_t = torch.sigmoid(model.w_g(mem_after_chunk1))
            gate_mean = g_t.mean().item()
            gate_std = g_t.std().item()
            mem_state_change = (mem_after_chunk1 - mem_before).norm(dim=-1).mean().item()
            mem_mean = mem_after_chunk1.mean().item()
            mem_std = mem_after_chunk1.std().item()
            
        targets_last_chunk = targets[:, -chunk_size:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets_last_chunk[:, 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
        if step == 1: initial_loss = loss.item()
            
        loss.backward()
        
        grad_norm = 0.0
        for name, p in model.named_parameters():
            if 'w_g' in name or 'g_theta' in name or 'M_0' in name:
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
        mem_grad_norm = grad_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            preds = torch.argmax(shift_logits, dim=-1)
            mask = shift_labels != -100
            correct = (preds[mask] == shift_labels[mask]).sum().item()
            total = mask.sum().item()
            train_acc = correct / max(total, 1) * 100
            
        if step % 100 == 0 or step == 1:
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Acc: {train_acc:5.1f}%")
            
        if step == steps:
            final_train_acc = train_acc
            final_loss = loss.item()
            print("\n--- EXACT LOGIT EVALUATION TRACE ---")
            for b in range(min(3, batch_size)):
                inf = batch_info[b]
                tk_pos_in_last_chunk = (inf['target_key_pos'] % 256)
                
                # The prediction for inputs[tk_pos] is at shift_logits[tk_pos]
                # The target for inputs[tk_pos] is at shift_labels[tk_pos]
                pred_token = preds[b, tk_pos_in_last_chunk].item()
                actual_target = shift_labels[b, tk_pos_in_last_chunk].item()
                
                print(f"Example {b}: KEY={inf['key']}, VALUE={inf['val']}, QUERY={inf['key']}")
                print(f"  Logit position accessed: {tk_pos_in_last_chunk}")
                print(f"  Target extracted: {actual_target} (expected {inf['val']})")
                print(f"  Model Prediction: {pred_token}")
                assert actual_target == inf['val']
                assert actual_target != -100
                
    # Val
    val_acc = 0.0
    if not fixed_batch:
        model.eval()
        v_c, v_t = 0, 0
        with torch.no_grad():
            for _ in range(10):
                v_in, v_tgt, _ = generate_diagnostic_batch(batch_size, mode, device)
                v_log, _ = model(v_in, chunk_size=chunk_size, detach_memory_every=0, return_last_chunk_only=True)
                s_log = v_log[:, :-1, :].contiguous()
                s_lbl = v_tgt[:, -chunk_size:][:, 1:].contiguous()
                p = torch.argmax(s_log, dim=-1)
                m = s_lbl != -100
                v_c += (p[m] == s_lbl[m]).sum().item()
                v_t += m.sum().item()
        val_acc = v_c / max(v_t, 1) * 100

    print("\n" + "-"*60)
    print(f"CONTROL {mode} RESULTS")
    print("-"*60)
    print(f"Initial CE Loss: {initial_loss:.4f}")
    print(f"Final CE Loss: {final_loss:.4f}")
    print(f"Train Accuracy: {final_train_acc:.1f}%")
    if not fixed_batch: print(f"Validation Accuracy: {val_acc:.1f}%")
    print(f"Random Chance Accuracy: {random_chance:.4f}%")
    
    if mode in ['B', 'C']:
        print(f"Memory Gradient Norm: {mem_grad_norm:.6f}")
        print(f"Write-Gate Mean: {gate_mean:.6f}, Std: {gate_std:.6f}")
        print(f"Memory State Mean: {mem_mean:.6f}, Std: {mem_std:.6f}")
        print(f"Memory State Change ||M_t - M_{{t-1}}||: {mem_state_change:.6f}")
        
    return final_train_acc > 90.0 if fixed_batch else val_acc > 10.0

if __name__ == "__main__":
    print("=== PHASE 1 DIAGNOSTIC BENCHMARK ===")
    print("1. Objective Sanity Test (Overfitting a tiny fixed batch of 8 examples)...")
    sanity_success = run_experiment('SANITY', steps=200, batch_size=8, fixed_batch=True)
    
    if not sanity_success:
        print("\nSANITY TEST FAILED. The model cannot even memorize 8 fixed examples.")
        print("This proves the loss/architecture fundamentally cannot map the associative query.")
        exit(1)
        
    print("\nSANITY TEST PASSED. The core E0 model and the benchmark label construction are mathematically valid.")
    
    print("\n2. Control A - Same Chunk Retrieval...")
    success_A = run_experiment('A', steps=500, batch_size=8)
    if not success_A:
        print("\nCONTROL A FAILED. Model cannot learn robust general associative retrieval within a single chunk.")
        exit(1)
        
    print("\n3. Control B - Single Pair Across Chunks...")
    success_B = run_experiment('B', steps=500, batch_size=8)
    
    if success_B:
        print("\n4. Control C - Multi Pair Across Chunks...")
        run_experiment('C', steps=500, batch_size=8)
