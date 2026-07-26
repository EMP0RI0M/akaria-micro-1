import torch
from chm.config import CHMConfig
from chm.fluxvm.adapter import FluxVMLatentAdapter

def test_stable_trajectory():
    cfg = CHMConfig(config_type="E", flux_mode="CONTROL")
    adapter = FluxVMLatentAdapter(cfg)
    
    # Stable tensor
    Y = torch.ones(2, 512, 4, 384)
    Y_damped, mem, metrics = adapter(Y, 0)
    
    assert metrics["D_global"] < 1e-5
    assert torch.allclose(metrics["beta"], torch.ones_like(metrics["beta"]))

def test_stream_instability():
    cfg = CHMConfig(config_type="E", flux_mode="CONTROL")
    adapter = FluxVMLatentAdapter(cfg)
    
    # Base stable tensor
    Y = torch.ones(2, 512, 4, 384)
    # Make one stream diverge heavily
    Y[:, :, 1, :] += 5.0
    
    Y_damped, mem, metrics = adapter(Y, 0)
    assert metrics["D_stream"] > 0
    assert torch.any(metrics["beta"] < 1.0) # Intervention should trigger

def test_recovery_trajectory():
    cfg = CHMConfig(config_type="E", flux_mode="CONTROL", gamma=0.5, alpha=1.0)
    adapter = FluxVMLatentAdapter(cfg)
    
    Y_stable = torch.ones(2, 512, 4, 384)
    Y_spike = torch.randn(2, 512, 4, 384) * 10
    
    # Step 1: Spike
    _, mem, metrics = adapter(Y_spike, 0)
    assert torch.any(mem > 0)
    
    # Step 2: Recovery (Stable)
    _, mem2, metrics2 = adapter(Y_stable, 1, prev_memory=mem)
    assert torch.all(mem2 < mem) # Memory must decay
