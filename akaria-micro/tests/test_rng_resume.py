import torch
import pytest
import os

def test_rng_state_resume():
    # Simulate saving CPU rng state
    orig_cpu_rng = torch.get_rng_state()
    
    # Save dummy checkpoint
    ckpt = {
        "cpu_rng_state": orig_cpu_rng.tolist(), # Convert to list to simulate JSON-like or arbitrary format load that drops ByteTensor
    }
    
    # Load and convert (mimicking train_tinystories logic)
    cpu_state = ckpt["cpu_rng_state"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        cpu_state = torch.tensor(cpu_state, dtype=torch.uint8, device='cpu')
        
    torch.set_rng_state(cpu_state.cpu())
    
    # Verify the states are identical
    new_cpu_rng = torch.get_rng_state()
    assert torch.all(orig_cpu_rng == new_cpu_rng)

    # CUDA
    if torch.cuda.is_available():
        orig_cuda_rng = torch.cuda.get_rng_state()
        
        ckpt["cuda_rng_state"] = orig_cuda_rng.tolist()
        
        cuda_state = ckpt["cuda_rng_state"]
        if not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8:
            cuda_state = torch.tensor(cuda_state, dtype=torch.uint8, device='cpu')
            
        torch.cuda.set_rng_state(cuda_state.cpu())
        
        new_cuda_rng = torch.cuda.get_rng_state()
        assert torch.all(orig_cuda_rng == new_cuda_rng)
