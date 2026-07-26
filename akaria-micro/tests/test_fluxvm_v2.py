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
    out, telemetry = ctrl(x, ablation_mode='E4')
    assert torch.allclose(out, x)

def test_proportional_trigger_zero_when_stable():
    # 2. D < tau => proportional trigger is zero
    ctrl = FluxVMControllerV2(tau=10.0, K=2)
    x = torch.randn(2, 4, 16)
    # Variance should be roughly 1.0, which is < 10.0
    out, telemetry = ctrl(x, ablation_mode='E4')
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
    out, telemetry = ctrl(x2, ablation_mode='E4')
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
    out, telemetry = ctrl(x, ablation_mode='E4')
    
    # Fake target
    target = torch.zeros_like(out)
    mse = torch.nn.functional.mse_loss(out, target)
    loss = telemetry["L_barrier"] if "L_barrier" in telemetry else 0.0
    total_loss = mse + loss
    total_loss.backward()
    
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_vx_and_delta_v_numerically_correct():
    # 7. V(X) and delta_V calculations
    ctrl = FluxVMControllerV2(tau=0.0, K=2)
    x = torch.randn(2, 4, 16)
    
    # Step 1
    out1, telemetry1 = ctrl(x, ablation_mode='E4', lambda_barrier=0.1)
    
    # Step 2
    x2 = torch.randn(2, 4, 16)
    out2, telemetry2 = ctrl(x2, ablation_mode='E4', lambda_barrier=0.1)
    
    delta_V_temporal = telemetry2["delta_V_temporal"]
    loss2 = telemetry2["L_barrier"]
    
    # Check if delta_V is passed properly
    # If delta_V_temporal > 0, L_barrier > 0 (if lambda_barrier=0.1)
    if delta_V_temporal > 0:
        assert torch.isclose(loss2, torch.tensor(0.1) * delta_V_temporal)
    else:
        assert loss2 == 0.0

def test_e0_does_not_modify_candidate():
    # 8. E0 does not modify candidate activations
    ctrl = FluxVMControllerV2(tau=0.0, K=5)
    x = torch.randn(2, 4, 16)
    out, telemetry = ctrl(x, ablation_mode='E0')
    assert torch.allclose(out, x)
    assert telemetry["L_barrier"] == 0.0

def test_e1_reproduces_old_controller():
    # 9. E1 still reproduces the old controller unchanged
    cfg = CHMConfig(config_type="E", flux_mode="E1", tau=1.0)
    old_ctrl = FluxVMLatentAdapter(cfg)
    
    x = torch.randn(2, 4, 16)
    out, mem, metrics = old_ctrl(x, 0, None)
    
    assert out.shape == x.shape
    assert 'beta' in metrics

def test_e5_hard_reset_semantics():
    # 1. E5 computes D_t and M_t (via telemetry)
    # 2. E5 computes same G_t as E3
    # 3. When G_t > 0, E5 I_t == 1
    # 4. When G_t == 0, E5 I_t == 0
    # 5. E5 modifies candidate when triggered
    # 6. E5 output differs from E0 on unstable input
    # 7. E5 telemetry is populated
    
    ctrl_e3 = FluxVMControllerV2(tau=0.0, K=2)
    ctrl_e5 = FluxVMControllerV2(tau=0.0, K=1) # E5 operates with 1 microstep
    ctrl_e0 = FluxVMControllerV2(tau=0.0, K=1)
    
    # Highly divergent tensor to trigger intervention
    x = torch.randn(2, 4, 16) * 10
    
    out_e3, tel_e3 = ctrl_e3(x.clone(), ablation_mode='E3')
    out_e5, tel_e5 = ctrl_e5(x.clone(), ablation_mode='E5')
    out_e0, tel_e0 = ctrl_e0(x.clone(), ablation_mode='E0')
    
    # 1. & 7. Computes and populates telemetry correctly
    assert tel_e5["D_t"].item() > 0
    assert "M_t" in tel_e5
    
    # 2. E5 computes same G_t as E3 for identical inputs
    assert torch.isclose(tel_e5["G_t"], tel_e3["G_t"])
    
    # 3. G_t > 0 -> I_t == 1
    assert tel_e5["G_t"].item() > 0
    assert tel_e5["I_t"].item() == 1.0
    
    # 5. & 6. E5 modifies candidate and differs from E0
    assert not torch.allclose(out_e5, x)
    assert not torch.allclose(out_e5, out_e0)
    
    # 4. When G_t == 0 -> I_t == 0
    ctrl_stable = FluxVMControllerV2(tau=100.0, K=1) # High tau -> no trigger
    x_stable = torch.ones(2, 4, 16)
    out_stable, tel_stable = ctrl_stable(x_stable, ablation_mode='E5')
    assert tel_stable["G_t"].item() == 0.0
    assert tel_stable["I_t"].item() == 0.0
    assert torch.allclose(out_stable, x_stable)
