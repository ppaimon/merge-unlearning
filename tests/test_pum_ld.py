# tests/test_pum_ld.py
# Tiny tests for PUM–LD invariants, runnable on toy or tiny LLaMA.
# Usage:
#   pytest -q tests/test_pum_ld.py
#   PUM_TEST_MODEL=llama pytest -q tests/test_pum_ld.py

from __future__ import annotations
import os
import math
from typing import Dict

import torch
import pytest

# Ensure <repo>/src is importable so package name is "trainer"
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import PUM–LD pieces directly from the package "trainer"
from trainer.unlearn.pum import (
    PUMTrainer, PUMConfig,
    compute_sigma_by_layer, generate_ld_noise_by_copy, ReparamPlan,
    tree_like, tree_add, tree_scale
)


# -------------------------------------------------------------------
# Toy LLaMA-like module (keys mimic HF LLaMA)
# -------------------------------------------------------------------
class ToyAttnMLPBlock(torch.nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        attn = torch.nn.Module()
        attn.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        attn.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        attn.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        attn.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.self_attn = attn

        mlp = torch.nn.Module()
        mlp.gate_proj = torch.nn.Linear(hidden, inter, bias=False)   # weight [inter, hidden]
        mlp.up_proj   = torch.nn.Linear(hidden, inter, bias=False)   # weight [inter, hidden]
        mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False)   # weight [hidden, inter]
        self.mlp = mlp

        self.input_layernorm = torch.nn.LayerNorm(hidden)
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden)


class ToyLlamaLike(torch.nn.Module):
    def __init__(self, n_layers=2, hidden=16, inter=32):
        super().__init__()
        wrapper = torch.nn.Module()
        layers = torch.nn.ModuleList([ToyAttnMLPBlock(hidden, inter) for _ in range(n_layers)])
        wrapper.layers = layers
        self.model = wrapper
        self.lm_head = torch.nn.Linear(hidden, hidden, bias=False)


def build_model(kind: str = "toy") -> torch.nn.Module:
    kind = (kind or "toy").lower()
    if kind != "llama":
        torch.manual_seed(0)
        return ToyLlamaLike(n_layers=2, hidden=16, inter=32).eval()
    # Try a real tiny LLaMA from 🤗; skip gracefully if unavailable
    try:
        from transformers import AutoModelForCausalLM
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        m.eval()
        return m
    except Exception as e:
        pytest.skip(f"Skipping real LLaMA: {e}. Using toy instead.")
        return ToyLlamaLike(n_layers=2, hidden=16, inter=32).eval()


# -------------------------------------------------------------------
# Test 1: Zero-sum basis for ε^0_k (before α scaling)
# -------------------------------------------------------------------
@pytest.mark.parametrize("model_kind", [os.getenv("PUM_TEST_MODEL", "toy")])
def test_zero_sum_basis(model_kind):
    model = build_model(model_kind)
    cfg = PUMConfig(m=6, alpha=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0], sigma_mode="rms_kappa", kappa=0.10,
                    reparam_attention_rotate=True, reparam_ffn_pair_permute=True, verbose=False)
    base_sd = model.state_dict()
    sigmas = compute_sigma_by_layer(base_sd, cfg)
    eps_by_copy = generate_ld_noise_by_copy(base_sd, sigmas, cfg)

    m = cfg.m
    # reconstruct ε^0_k = ε_k / α_k and check sum_k ε^0_k ≈ 0 per layer
    for name, t in base_sd.items():
        if t.dim() < 2:
            continue
        sum_eps0 = torch.zeros_like(t)
        denom = 1e-12
        for k in range(m):
            eps_k = eps_by_copy[k][name]
            eps0_k = eps_k / cfg.alpha[k]
            sum_eps0.add_(eps0_k)
            denom += eps0_k.norm().item()
        rel = sum_eps0.norm().item() / denom
        assert rel < 1e-6, f"Zero-sum basis violated for param {name}: rel={rel}"


# -------------------------------------------------------------------
# Test 2: T^{-1}(T(W)) ≈ W for sampled reparams
# -------------------------------------------------------------------
@pytest.mark.parametrize("model_kind", [os.getenv("PUM_TEST_MODEL", "toy")])
def test_reparam_inverse_identity(model_kind):
    model = build_model(model_kind)
    cfg = PUMConfig(m=4, sigma_mode="fixed", sigma_fixed=0.01, verbose=False)
    base_sd = model.state_dict()

    plan = ReparamPlan(model, cfg)
    T_fwd, T_inv = plan.sample_T(k=0, cfg=cfg)

    W = base_sd
    WT = T_fwd(W)
    W_rec = T_inv(WT)

    max_abs = 0.0
    for k in W.keys():
        if k in W_rec:
            d = (W[k] - W_rec[k]).abs().max().item()
            max_abs = max(max_abs, d)
    assert max_abs < 1e-5, f"Reparam inverse not identity: max_abs={max_abs}"


# -------------------------------------------------------------------
# Test 3: Harmonic de-noising cancels α_k-correlated term (first-order)
# -------------------------------------------------------------------
@pytest.mark.parametrize("model_kind", [os.getenv("PUM_TEST_MODEL", "toy")])
def test_harmonic_denoising_linearized(model_kind):
    torch.manual_seed(0)
    model = build_model(model_kind)
    cfg = PUMConfig(
        m=6, alpha=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        sigma_mode="rms_kappa", kappa=0.10,
        eta_srv=1.0, seed_noise=17, seed_reparam=23,
        reparam_attention_rotate=True, reparam_ffn_pair_permute=True,
        verbose=False
    )

    base_sd = model.state_dict()
    sigmas = compute_sigma_by_layer(base_sd, cfg)
    eps_by_copy = generate_ld_noise_by_copy(base_sd, sigmas, cfg)

    # Pre-sample T_k used by trainer (same seeds/config)
    plan = ReparamPlan(model, cfg)
    T_fwds, T_invs = [], []
    for k in range(cfg.m):
        T_fwd, T_inv = plan.sample_T(k, cfg)
        T_fwds.append(T_fwd); T_invs.append(T_inv)

    # Synthetic Δ* and "J≈γ I"
    delta_star = {k: 1e-3 * torch.randn_like(v) for k, v in base_sd.items()}
    gamma = 0.7

    call_idx = {"i": 0}
    def client_unlearn_fn(theta_pub_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        i = call_idx["i"]
        d = tree_add(delta_star, tree_scale(eps_by_copy[i], gamma))  # Δ* + γ ε_k (server/original)
        out = T_fwds[i](d)                                           # publish-space
        call_idx["i"] = i + 1
        return out

    trainer = PUMTrainer(model, cfg)
    new_state, bar_delta = trainer.run_round(client_unlearn_fn)

    max_rel = 0.0
    eps = 1e-12
    for k in delta_star.keys():
        num = (bar_delta[k] - delta_star[k]).norm().item()
        den = delta_star[k].norm().item() + eps
        max_rel = max(max_rel, num / den)
    assert max_rel < 1e-4, f"Harmonic de-noising failed: max_rel={max_rel}"
