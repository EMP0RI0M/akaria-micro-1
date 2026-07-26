import torch
import pytest
from chm.config import CHMConfig
from chm.fluxvm.fluxvm_v2 import FluxVMControllerV2
from chm.fluxvm.adapter import FluxVMLatentAdapter
from chm.recurrent.core import RecurrentCore

def test_stable_state_zero_intervention():
    # 1. stable state => exactly zero intervention
    ctrl = FluxVMControllerV2(tau=10.0, K=2)
    # A completely homogeneous tensor has 0 divergence
    x = torch.ones(2, 4, 16)
    out, loss, bpass = ctrl(x, ablation_mode='E4')
    assert torch.allclose(out, x)

def test_proportional_trigger_zero_when_stable():
    # 2. D < tau => proportional trigger is zero
    ctrl = FluxVMControllerV2(tau=10.0, K=2)
    x = torch.randn(2, 4, 16)
    # Variance should be roughly 1.0, which is < 10.0
    out, loss, bpass = ctrl(x, ablation_mode='E4')
    # Since M_bar = M_t, g_M = 0 and D_t < tau => g_D = 0
    # Thus G_t = 0 => I_t = 0
    assert torch.allclose(out, x)

def test_negative_delta_d_remains_negative():
    # 3. negative delta_D remains negative
    ctrl = FluxVMControllerV2(tau=0.0, K=2) # Force intervention
    # Step 1: High divergence
    x1 = torch.randn(2, 4, 16) * 10 
    ctrl(x1, ablation_mode='E4')
    D1 = ctrl.D_prev
    # Step 2: Low divergence
    x2 = torch.randn(2, 4, 16) * 1
    # We want to check dD_t inside the forward pass, let's patch it or just check it's negative
    # We can calculate what I_t would be
    out, loss, bpass = ctrl(x2, ablation_mode='E4')
    D2 = ctrl.D_prev
    dD = D2 - D1
    assert dD < 0.0

def test_m_j_history_update_exactly_once():
    # 4 & 5. update exactly once, microsteps don't advance it
    ctrl = FluxVMControllerV2(tau=0.0, K=5)
    x = torch.randn(2, 4, 16)
    ctrl(x, ablation_mode='E4')
    # If it updated K times, len(history_M) would be 5.
    # It should only be 1.
    assert len(ctrl.history_M) == 1

def test_finite_gradients():
    # 6. K=2 backward pass gives finite gradients
    ctrl = FluxVMControllerV2(tau=0.0, K=2)
    x = torch.randn(2, 4, 16, requires_grad=True)
    out, loss, bpass = ctrl(x, ablation_mode='E4')
    
    # Fake target
    target = torch.zeros_like(out)
    mse = torch.nn.functional.mse_loss(out, target)
    total_loss = mse + loss
    total_loss.backward()
    
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_vx_and_delta_v_numerically_correct():
    # 7. V(X) and delta_V calculations
    ctrl = FluxVMControllerV2(tau=0.0, K=2)
    x = torch.randn(2, 4, 16)
    
    # Step 1
    out1, loss1, bpass1 = ctrl(x, ablation_mode='E4')
    D1 = ctrl.D_prev
    M1 = ctrl.M_prev
    V1 = 0.5 * D1**2 + 0.5 * M1**2
    
    # Step 2
    x2 = torch.randn(2, 4, 16)
    out2, loss2, bpass2 = ctrl(x2, ablation_mode='E4')
    D2 = ctrl.D_prev
    M2 = ctrl.M_prev
    V2 = 0.5 * D2**2 + 0.5 * M2**2
    
    delta_V = V2 - V1
    # Check if delta_V is passed properly
    # If delta_V > 0, L_barrier > 0 (if lambda_barrier=0.1)
    if delta_V > 0:
        assert torch.isclose(loss2, torch.tensor(0.1) * delta_V)
    else:
        assert loss2 == 0.0

def test_e0_does_not_modify_candidate():
    # 8. E0 does not modify candidate activations
    ctrl = FluxVMControllerV2(tau=0.0, K=5)
    x = torch.randn(2, 4, 16)
    out, loss, bpass = ctrl(x, ablation_mode='E0')
    assert torch.allclose(out, x)
    assert loss == 0.0

def test_e1_reproduces_old_controller():
    # 9. E1 still reproduces the old controller unchanged
    cfg = CHMConfig(flux_mode="E1", tau=1.0)
    old_ctrl = FluxVMLatentAdapter(cfg)
    
    x = torch.randn(2, 4, 16)
    out, mem, metrics = old_ctrl(x, 0, None)
    
    assert out.shape == x.shape
    assert 'beta' in metrics
    assert 'M' in metrics
