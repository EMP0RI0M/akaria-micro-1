import torch
import pytest
from scripts.train_tinystories import safe_telemetry_to_tensor, check_nan

def test_safe_telemetry_to_tensor():
    # 1. Pure floats
    v1 = safe_telemetry_to_tensor([1.0, 2.0, 3.0])
    assert torch.allclose(v1, torch.tensor([1.0, 2.0, 3.0]))
    
    # 2. Mixed floats and tensors
    v2 = safe_telemetry_to_tensor([torch.tensor(1.0), 2.0, torch.tensor(3.0)])
    assert torch.allclose(v2, torch.tensor([1.0, 2.0, 3.0]))
    
    # 3. Requires grad tensor (should detach properly for diagnostic checking)
    t3 = torch.tensor(4.0, requires_grad=True)
    v3 = safe_telemetry_to_tensor([1.0, t3])
    assert torch.allclose(v3, torch.tensor([1.0, 4.0]))
    assert not v3.requires_grad
    
    # 4. Scalar float
    v4 = safe_telemetry_to_tensor(1.0)
    assert torch.allclose(v4, torch.tensor([1.0]))
    
def test_check_nan():
    # Should pass
    check_nan(torch.tensor([1.0, 2.0]), "Test", 0, "Contestant")
    
    # Should fail on NaN
    with pytest.raises(RuntimeError, match="NaN/Inf detected"):
        check_nan(torch.tensor([1.0, float('nan')]), "Test", 0, "Contestant")
        
    # Should fail on Inf
    with pytest.raises(RuntimeError, match="NaN/Inf detected"):
        check_nan(torch.tensor([float('inf'), 2.0]), "Test", 0, "Contestant")
