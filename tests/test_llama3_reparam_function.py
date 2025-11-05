# tests/test_llama3_reparam_function.py
import os
import sys
from pathlib import Path
import torch
import pytest

# Ensure `src/` is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trainer.unlearn.pum import PUMConfig, ReparamPlan  # noqa: E402

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None


MODEL_ID_DEFAULT = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
MODEL_ID = os.getenv("PUM_TEST_MODEL_ID", MODEL_ID_DEFAULT)

# Keep it lightweight: just reparam the first N layers (invariance holds per-layer)
MAX_LAYERS = int(os.getenv("PUM_TEST_LAYERS", "1"))

@pytest.mark.skipif(AutoModelForCausalLM is None or AutoTokenizer is None, reason="transformers not installed")
def test_llama3_reparam_function_invariance_subset():
    """
    Functional invariance test (no training):
      - Load Llama-3 model & tokenizer (CPU, fp32)
      - Get logits on a short prompt
      - Apply reparam T to first MAX_LAYERS layers only
      - Get logits again -> must match (within 1e-5 in fp32)
      - Apply inverse T^-1 on those layers only
      - Verify weights return to original (max_abs < 1e-5)
    """
    torch.set_grad_enabled(False)
    device = "cpu"

    # 1) Load model & tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    # some Llama checkpoints lack pad_token; set to eos for batching safety
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.eval()
    model.to(device)

    # 2) Prepare a tiny input
    text = "A quick test for PUM reparameterization invariance."
    enc = tok(text, return_tensors="pt")
    for k in enc:
        enc[k] = enc[k].to(device)

    with torch.no_grad():
        out_ref = model(**enc).logits.detach().clone()

    # 3) Build reparam plan (no training; just transformations)
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
        reparam_attention_rope_aware=True,  # NEW
        reparam_ffn_pair_permute=True,
        reparam_residual_permute=False,
        attn_num_heads=None,
        attn_num_kv_heads=None,
        verbose=False,
    )

    plan = ReparamPlan(model, cfg)

    # Restrict to first MAX_LAYERS for speed
    plan.attn_blocks = plan.attn_blocks[:MAX_LAYERS]
    plan.ffn_blocks = plan.ffn_blocks[:MAX_LAYERS]

    # 4) Sample T and T^-1 for a fixed "copy" k=0
    T_fwd, T_inv = plan.sample_T(k=0, cfg=cfg)

    # Build a minimal SD with only the keys we touch
    full_sd = model.state_dict()
    needed = set()
    for blk in plan.attn_blocks:
        needed.update([blk["q"], blk["k"], blk["v"], blk["o"]])
    for blk in plan.ffn_blocks:
        needed.update([blk["gate"], blk["up"], blk["down"]])

    W0 = {k: full_sd[k].clone() for k in needed}

    # 5) Apply forward reparam (T) to those keys and load them into the model
    W_T = T_fwd(W0)
    missing, unexpected = model.load_state_dict(W_T, strict=False)
    # Not all keys provided; 'missing' is expected. 'unexpected' should be empty.
    assert len(unexpected) == 0, f"Unexpected keys when loading reparam: {unexpected}"

    # 6) Forward pass must match the reference logits (orthogonal invariance)
    with torch.no_grad():
        out_reparam = model(**enc).logits.detach().clone()

    # fp32 => very small tolerance
    max_abs_logits = (out_ref - out_reparam).abs().max().item()
    assert max_abs_logits < 1e-5, f"Functional invariance failed: max|Δlogits|={max_abs_logits:.3e}"

    # 7) Apply inverse on those keys and verify weights are restored
    #    Grab state dict again (includes the reparammed tensors), then invert
    full_sd_after = model.state_dict()
    W_now = {k: full_sd_after[k].clone() for k in needed}
    W_rec = T_inv(W_now)

    max_abs_w = 0.0
    worst_k = None
    for k in needed:
        diff = (W0[k] - W_rec[k]).abs().max().item()
        if diff > max_abs_w:
            max_abs_w = diff
            worst_k = k

    assert max_abs_w < 1e-5, f"Inverse did not restore weights (max|Δ|={max_abs_w:.3e}, key={worst_k})"

# pytest -q tests/test_llama3_reparam_function.py::test_llama3_reparam_function_invariance_subset
