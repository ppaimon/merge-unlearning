# tests/test_llama3_reparam_inverse.py
import os
import torch
import pytest

# The reparam plan + config from your PUM engine
from trainer.unlearn.pum import PUMConfig, ReparamPlan

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    AutoModelForCausalLM = None


MODEL_ID_DEFAULT = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
MODEL_ID = os.getenv("PUM_TEST_MODEL_ID", MODEL_ID_DEFAULT)

# Limit how many layers we reparam to keep this "quick"
MAX_LAYERS = int(os.getenv("PUM_TEST_LAYERS", "1"))

# Allow a full-model check if explicitly requested
RUN_FULL = os.getenv("PUM_TEST_FULL", "0") == "1"


@pytest.mark.skipif(AutoModelForCausalLM is None, reason="transformers not installed")
def test_llama3_reparam_inverse_identity_subset():
    """
    Quick identity test on Llama-3.2-1B-Instruct weights:
    - Load model (CPU, fp32)
    - Build ReparamPlan (will infer H_Q, H_KV from model.config)
    - Restrict to the first MAX_LAYERS layers for speed/memory
    - Sample T (S_KV, U) with fixed seed; apply T and T^{-1}
    - Check max |W - W_rec| across touched params
    """
    torch.set_grad_enabled(False)

    # Load model on CPU in fp32 to keep numerical error tiny
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    # Minimal PUM config for reparam only (no training)
    cfg = PUMConfig(
        m=2,
        alpha=[1.0, 1.5],
        sigma_mode="fixed",
        sigma_fixed=0.01,
        kappa=0.10,
        eta_srv=1.0,
        seed_train=0,
        seed_noise=17,
        seed_reparam=23,
        reparam_attention_rotate=True,
        reparam_ffn_pair_permute=True,
        reparam_residual_permute=False,
        attn_num_heads=None,
        attn_num_kv_heads=None,
        verbose=False,
    )

    plan = ReparamPlan(model, cfg)

    # Restrict to a few layers to be lightweight
    plan.attn_blocks = plan.attn_blocks[:MAX_LAYERS]
    plan.ffn_blocks = plan.ffn_blocks[:MAX_LAYERS]

    # Get closures (they capture S_KV, U, P) for this "copy" k=0
    T_fwd, T_inv = plan.sample_T(k=0, cfg=cfg)

    # Build a minimal state dict containing only the keys we will touch
    full_sd = model.state_dict()
    needed_keys = set()
    for blk in plan.attn_blocks:
        needed_keys.update([blk["q"], blk["k"], blk["v"], blk["o"]])
    for blk in plan.ffn_blocks:
        needed_keys.update([blk["gate"], blk["up"], blk["down"]])

    W = {k: full_sd[k].clone() for k in needed_keys}

    # Apply reparam forward and inverse on the subset
    WT = T_fwd(W)
    W_rec = T_inv(WT)

    # Compare tensors
    max_abs = 0.0
    worst_k = None
    for k in needed_keys:
        diff = (W[k] - W_rec[k]).abs().max().item()
        if diff > max_abs:
            max_abs = diff
            worst_k = k

    # fp32 should be very tight
    assert max_abs < 1e-5, f"Max abs diff {max_abs:.3e} on key {worst_k}"


@pytest.mark.skipif(AutoModelForCausalLM is None, reason="transformers not installed")
@pytest.mark.skipif(not RUN_FULL, reason="Set PUM_TEST_FULL=1 to enable the full-model identity check")
def test_llama3_reparam_inverse_identity_full_model():
    """
    Full-model identity test. This will touch all layers and allocate extra memory.
    Enable with: PUM_TEST_FULL=1 pytest -q tests/test_llama3_reparam_inverse.py::test_llama3_reparam_inverse_identity_full_model
    """
    torch.set_grad_enabled(False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    cfg = PUMConfig(
        m=2,
        alpha=[1.0, 1.5],
        sigma_mode="fixed",
        sigma_fixed=0.01,
        kappa=0.10,
        eta_srv=1.0,
        seed_train=0,
        seed_noise=17,
        seed_reparam=23,
        reparam_attention_rotate=True,
        reparam_ffn_pair_permute=True,
        reparam_residual_permute=False,
        attn_num_heads=None,
        attn_num_kv_heads=None,
        verbose=False,
    )

    plan = ReparamPlan(model, cfg)
    T_fwd, T_inv = plan.sample_T(k=0, cfg=cfg)

    # To avoid deep-copying the *entire* 1B-param state dict, we still only pass
    # the keys the plan will touch (but for ALL layers).
    full_sd = model.state_dict()
    needed_keys = set()
    for blk in plan.attn_blocks:
        needed_keys.update([blk["q"], blk["k"], blk["v"], blk["o"]])
    for blk in plan.ffn_blocks:
        needed_keys.update([blk["gate"], blk["up"], blk["down"]])

    W = {k: full_sd[k].clone() for k in needed_keys}

    WT = T_fwd(W)
    W_rec = T_inv(WT)

    max_abs = 0.0
    worst_k = None
    for k in needed_keys:
        diff = (W[k] - W_rec[k]).abs().max().item()
        if diff > max_abs:
            max_abs = diff
            worst_k = k

    assert max_abs < 1e-5, f"Max abs diff {max_abs:.3e} on key {worst_k}"
