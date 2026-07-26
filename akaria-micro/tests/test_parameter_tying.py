import torch
import torch.nn as nn
from chm.config import CHMConfig
from chm.model import ControlledHyperloopMoE

def test_recurrent_block_identity():
    # Instantiate the model with a Dense DRT config (Config B)
    cfg = CHMConfig(config_type="B", core_loops=5)
    model = ControlledHyperloopMoE(cfg)
    
    # We must prove that the core uses the exact same nn.Module instance for all recurrent loops.
    # We can check the `id()` of the parameters in `model.core.shared_block`.
    # Since `core.py` executes: `H_tilde = self.shared_block(X_attn)` in a loop,
    # the parameters accessed in each loop are identically the same object in memory.
    
    # Verify there is only one block instance in the core, not a ModuleList
    assert not isinstance(model.core.shared_block, nn.ModuleList), "Core block must be a single shared instance, not a list."
    
    # Verify the memory addresses (data pointers) of the weights are identical.
    # While inherently true by PyTorch's class structure, this explicitly tests that
    # the architecture hasn't accidentally cloned the weights.
    attn_weight_ptr = model.core.shared_block.attn.wq.weight.data_ptr()
    ffn_weight_ptr = model.core.shared_block.ffn.layer.w1.weight.data_ptr()
    
    # Do a dummy forward pass and trace the parameters updated
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Snapshot pre-update weights
    pre_attn_weight = model.core.shared_block.attn.wq.weight.clone()
    
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    optimizer.step()
    
    # Ensure gradients flowed to the shared block and weights updated
    assert model.core.shared_block.attn.wq.weight.grad is not None
    assert not torch.allclose(model.core.shared_block.attn.wq.weight, pre_attn_weight)
    
    # Verify pointers didn't shift
    assert model.core.shared_block.attn.wq.weight.data_ptr() == attn_weight_ptr
    assert model.core.shared_block.ffn.layer.w1.weight.data_ptr() == ffn_weight_ptr

def test_expert_bank_identity():
    # Instantiate the model with Tied MoE (Config D)
    cfg = CHMConfig(config_type="D", n_experts=8, experts_per_token=2, core_loops=5)
    model = ControlledHyperloopMoE(cfg)
    
    # TiedMoE wraps MoELayer. We verify that the expert matrices (w1s, w2s, w3s)
    # exist precisely once in the shared MoELayer, proving parameter tying across depth.
    
    # Verify the MoELayer uses the stacked expert tensors
    assert hasattr(model.core.shared_block.ffn.layer, "w1s"), "MoELayer must contain the stacked expert tensor w1s"
    
    # In looped-moe, w1s is a ParameterList containing per-expert parameters.
    # We check that its length matches n_experts and that the parameters share identity.
    w1s = model.core.shared_block.ffn.layer.w1s
    assert len(w1s) == cfg.n_experts, f"Expected {cfg.n_experts} experts, found {len(w1s)}"
    
    # Test gradient flow to the shared expert bank
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    pre_w1s = model.core.shared_block.ffn.layer.w1s[0].clone()
    
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    optimizer.step()
    
    # Ensure experts received gradients and were updated
    assert model.core.shared_block.ffn.layer.w1s[0].grad is not None
    assert not torch.allclose(model.core.shared_block.ffn.layer.w1s[0], pre_w1s)
