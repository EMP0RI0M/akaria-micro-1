import torch
import torch.nn as nn
import time
from chm.config import CHMConfig
from chm.memory import E0Memory
from chm.blocks import TransformerBlock
from chm.normalization import RMSNorm

class BaselineTransformer(nn.Module):
    """Standard dense Transformer as a positive control for the benchmark."""
    def __init__(self, cfg: CHMConfig, num_layers=7):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(num_layers)])
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        return self.lm_head(x), {} # Return empty telemetry to match API

def generate_ladder_batch(batch_size, ladder_level, device='cpu'):
    """
    Ladder Levels:
    'A0': Direct copy. (No keys, just VALUE -> QUERY_MARKER -> predict VALUE)
    'A1': 1 KV pair
    'A2': 2 KV pairs
    'A3': 4 KV pairs
    'A4': 8 KV pairs
    All sequences are 256 tokens.
    """
    seq_len = 256
    inputs = torch.randint(3001, 16000, (batch_size, seq_len), device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    query_marker = 999
    info = []
    
    for b in range(batch_size):
        if ladder_level == 'A0':
            val = torch.randint(1000, 3000, (1,)).item()
            pos = torch.randint(0, 100, (1,)).item()
            inputs[b, pos] = val
            
            query_pos = seq_len - 3
            inputs[b, query_pos] = query_marker
            targets[b, query_pos + 1] = val
            
            info.append({
                'key': None, 'val': val,
                'query_marker_pos': query_pos,
                'target_key_pos': query_pos, # For A0, prediction is immediately after query marker
                'target_val_pos': query_pos + 1
            })
            
        else:
            num_pairs = int(ladder_level[1:])
            keys = torch.randperm(2000)[:num_pairs] + 1000
            values = torch.randperm(2000)[:num_pairs] + 1000
            
            positions = torch.randperm(150)[:num_pairs]
            for i in range(num_pairs):
                pos = positions[i].item()
                inputs[b, pos] = keys[i]
                inputs[b, pos+1] = values[i]
                
            target_idx = torch.randint(0, num_pairs, (1,)).item()
            target_key = keys[target_idx]
            target_val = values[target_idx]
            
            query_pos = seq_len - 3
            inputs[b, query_pos] = query_marker
            inputs[b, query_pos + 1] = target_key
            targets[b, query_pos + 2] = target_val
            
            info.append({
                'key': target_key.item(), 'val': target_val.item(),
                'query_marker_pos': query_pos,
                'target_key_pos': query_pos + 1,
                'target_val_pos': query_pos + 2
            })
            
    return inputs, targets, info

def verify_and_print_batch(inputs, targets, batch_info, ladder_level):
    print(f"\n--- EXPLICIT ALIGNMENT CHECK: {ladder_level} ---")
    b = 0
    inf = batch_info[b]
    qm_pos = inf['query_marker_pos']
    tk_pos = inf['target_key_pos']
    tv_pos = inf['target_val_pos']
    
    assert inputs[b, qm_pos] == 999
    if ladder_level != 'A0':
        assert inputs[b, tk_pos] == inf['key']
    assert targets[b, tv_pos] == inf['val']
    assert targets[b, tv_pos] != -100
    
    print(f"Example 0:")
    if ladder_level == 'A0':
        print(f"  VALUE = {inf['val']}")
    else:
        print(f"  KEY = {inf['key']}  VALUE = {inf['val']}")
    print(f"  query_marker_pos = {qm_pos}, token = {inputs[b, qm_pos]}")
    print(f"  target_pred_pos  = {tk_pos}, token = {inputs[b, tk_pos]}")
    print(f"  target_val_pos   = {tv_pos}, expected VALUE = {inf['val']}")
    print(f"  actual supervised target at target_val_pos = {targets[b, tv_pos]}")
    print("Alignment strictly verified.\n")

def top_k_accuracy(preds_logits, labels, k=5):
    """Calculate top-k accuracy."""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    
    valid_logits = preds_logits[mask]
    valid_labels = labels[mask]
    
    top_k_preds = torch.topk(valid_logits, k, dim=-1).indices
    correct = (top_k_preds == valid_labels.unsqueeze(-1)).any(dim=-1)
    
    return (correct.sum().item() / max(mask.sum().item(), 1)) * 100

def run_ladder_stage(model, optimizer, cfg, ladder_level, steps=2000, batch_size=32):
    print("\n" + "="*80)
    print(f"RUNNING LADDER {ladder_level} | Model: {model.__class__.__name__}")
    print("="*80)
    
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Verify alignment
    v_inputs, v_targets, v_info = generate_ladder_batch(batch_size, ladder_level, device)
    verify_and_print_batch(v_inputs, v_targets, v_info, ladder_level)
    
    consecutive_success = 0
    final_val_acc = 0.0
    
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        
        inputs, targets, batch_info = generate_ladder_batch(batch_size, ladder_level, device)
        
        # Determine if model is E0Memory or BaselineTransformer
        if isinstance(model, E0Memory):
            # For same-chunk, we only use the first 256 chunk.
            logits, telemetry = model(inputs, chunk_size=256, detach_memory_every=0, return_last_chunk_only=True)
        else:
            logits, telemetry = model(inputs)
            
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                preds = torch.argmax(shift_logits, dim=-1)
                mask = shift_labels != -100
                correct = (preds[mask] == shift_labels[mask]).sum().item()
                total = mask.sum().item()
                train_acc = correct / max(total, 1) * 100
                top5_acc = top_k_accuracy(shift_logits, shift_labels, k=5)
            
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Top-1: {train_acc:5.1f}% | Top-5: {top5_acc:5.1f}%")
            
            # Print example from training batch
            if step == 1 or step % 500 == 0:
                b = 0
                inf = batch_info[b]
                tk_pos = inf['target_key_pos']
                pred_token = preds[b, tk_pos].item()
                actual_target = shift_labels[b, tk_pos].item()
                print(f"  -> Example Trace: KEY={inf['key']} VALUE={inf['val']} QUERY={inf['key'] if inf['key'] else 'MARKER'} PRED={pred_token} TGT={actual_target}")
                
            # Validation Evaluation
            model.eval()
            val_c1, val_c5, val_t = 0, 0, 0
            with torch.no_grad():
                for _ in range(5):
                    v_in, v_tgt, _ = generate_ladder_batch(batch_size, ladder_level, device)
                    if isinstance(model, E0Memory):
                        v_log, _ = model(v_in, chunk_size=256, detach_memory_every=0, return_last_chunk_only=True)
                    else:
                        v_log, _ = model(v_in)
                        
                    s_log = v_log[:, :-1, :].contiguous()
                    s_lbl = v_tgt[:, 1:].contiguous()
                    
                    p = torch.argmax(s_log, dim=-1)
                    m = s_lbl != -100
                    val_c1 += (p[m] == s_lbl[m]).sum().item()
                    val_t += m.sum().item()
                    
                    # Top-5 logic accumulated
                    valid_log = s_log[m]
                    valid_lbl = s_lbl[m]
                    if valid_log.size(0) > 0:
                        top_k_preds = torch.topk(valid_log, 5, dim=-1).indices
                        val_c5 += (top_k_preds == valid_lbl.unsqueeze(-1)).any(dim=-1).sum().item()
                        
            val_acc1 = val_c1 / max(val_t, 1) * 100
            val_acc5 = val_c5 / max(val_t, 1) * 100
            final_val_acc = val_acc1
            print(f"         | VAL Top-1: {val_acc1:5.1f}% | VAL Top-5: {val_acc5:5.1f}%")
            
            if val_acc1 >= 95.0:
                consecutive_success += 1
                if consecutive_success >= 3:
                    print(f"\n[SUCCESS] {model.__class__.__name__} passed {ladder_level} early at step {step}.")
                    return True
            else:
                consecutive_success = 0
                
    print(f"\n[COMPLETED] {model.__class__.__name__} finished {ladder_level} with final Val Top-1: {final_val_acc:.1f}%")
    return final_val_acc >= 90.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    cfg_base = CHMConfig(
        vocab_size=16384, d_model=384, n_heads=6, 
        prelude_layers=1, core_loops=5, coda_layers=1, 
        config_type="E", sequence_length=512
    )
    
    # Configure the dense baseline
    cfg_dense = CHMConfig(
        vocab_size=16384, d_model=384, n_heads=6,
        n_experts=1, experts_per_token=1, d_ff=1024, # Dense config
        sequence_length=512
    )
    
    model_baseline = BaselineTransformer(cfg_dense, num_layers=7).to(device)
    model_e0 = E0Memory(cfg_base, num_mem_tokens=16, mem_refinement_steps=1).to(device)
    
    print(f"Baseline Params: {sum(p.numel() for p in model_baseline.parameters()) / 1e6:.2f}M")
    print(f"E0-Memory Params: {sum(p.numel() for p in model_e0.parameters()) / 1e6:.2f}M")
    
    ladders = ['A0', 'A1', 'A2', 'A3', 'A4']
    
    for level in ladders:
        # Reset optimizers per stage to ensure pristine learning dynamics
        opt_base = torch.optim.AdamW(model_baseline.parameters(), lr=1e-3, weight_decay=0.01)
        opt_e0 = torch.optim.AdamW(model_e0.parameters(), lr=1e-3, weight_decay=0.01)
        
        # Test Baseline Transformer
        success_base = run_ladder_stage(model_baseline, opt_base, cfg_dense, level, steps=2000, batch_size=32)
        
        # Test E0-Memory
        success_e0 = run_ladder_stage(model_e0, opt_e0, cfg_base, level, steps=2000, batch_size=32)
        
        print("\n" + "*"*80)
        print(f"STAGE {level} RESULTS SUMMARY")
        print(f"Baseline Transformer: {'PASSED' if success_base else 'FAILED'}")
        print(f"E0-Memory:          {'PASSED' if success_e0 else 'FAILED'}")
        print("*"*80 + "\n")
        
        if not success_base and not success_e0:
            print(f"BOTH models failed {level}. The task/benchmark might be broken or too hard. Stopping.")
            break
            
        if success_base and not success_e0:
            print(f"Transformer succeeded but E0-Memory failed at {level}.")
            print("This isolates the failure to the E0 architecture's associative capacity.")
            break

if __name__ == "__main__":
    main()
