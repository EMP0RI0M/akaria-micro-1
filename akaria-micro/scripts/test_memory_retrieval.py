import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.memory import E0Memory

def generate_retrieval_batch(batch_size, logical_seq_len, vocab_size=16384, device='cpu'):
    """
    Generates a synthetic associative retrieval task.
    KEY is placed near the beginning.
    QUERY is placed near the end.
    The model must output the VALUE associated with the KEY.
    """
    # Create random filler
    inputs = torch.randint(1000, 5000, (batch_size, logical_seq_len), device=device)
    targets = torch.full((batch_size, logical_seq_len), -100, dtype=torch.long, device=device)
    
    query_marker = 999
    
    for b in range(batch_size):
        # Random keys and values
        key = torch.randint(100, 500, (1,)).item()
        val = torch.randint(501, 900, (1,)).item()
        
        # Place key-value pair in the first chunk (e.g. index 10, 11)
        inputs[b, 10] = key
        inputs[b, 11] = val
        
        # Place query at the end
        query_pos = logical_seq_len - 3
        inputs[b, query_pos] = query_marker
        inputs[b, query_pos + 1] = key
        
        # Target for the position after the key is the value
        targets[b, query_pos + 1] = val
        
    return inputs, targets

def main():
    print("="*60)
    print("LONG-RANGE MEMORY BENCHMARK - SYNTHETIC RETRIEVAL")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    contexts = [512, 1024, 2048, 4096, 8192]
    batch_size = 4
    chunk_size = 256
    
    for ctx_len in contexts:
        print(f"\nTesting Logical Context Length: {ctx_len}")
        print("-" * 40)
        
        for step in range(10): # Tiny pilot steps
            optimizer.zero_grad()
            
            inputs, targets = generate_retrieval_batch(batch_size, ctx_len, device=device)
            
            # Forward pass with memory chunking
            logits, telemetry = model(inputs, chunk_size=chunk_size, detach_memory_every=4)
            
            # Loss calculation
            # Shift logits and targets by 1 for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))
            
            if step == 0:
                print(f"  Step {step} Initial Loss: {loss.item():.4f}")
                
            loss.backward()
            optimizer.step()
            
            if step == 9:
                print(f"  Step {step} Final Loss:   {loss.item():.4f}")
                
                # Check accuracy
                with torch.no_grad():
                    preds = torch.argmax(shift_logits, dim=-1)
                    mask = shift_labels != -100
                    correct = (preds[mask] == shift_labels[mask]).sum().item()
                    total = mask.sum().item()
                    acc = correct / max(total, 1) * 100
                    print(f"  Retrieval Accuracy: {acc:.1f}%")

if __name__ == "__main__":
    main()
