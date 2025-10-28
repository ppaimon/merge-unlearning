# src/trainer/unlearn/pum.py
# PUM–LD: No clipping; fixed σ; α-scaled linearly dependent copy noise; T_k reparams; harmonic de-noise
# Author: (your name)
# License: MIT (same as repo)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Tuple
import math
import hashlib
import logging
import random

import torch
from torch import nn

logger = logging.getLogger(__name__)


# -----------------------------
# Utility: tree ops over state_dict-like structures
# -----------------------------
def tree_like(x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in x.items()}

def tree_zeros_like(x: Mapping[str, torch.Tensor], device=None, dtype=None) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in x.items():
        out[k] = torch.zeros_like(v, device=device or v.device, dtype=dtype or v.dtype)
    return out

def tree_add(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in a:
        if k in b:
            out[k] = a[k] + b[k]
    return out

def tree_sub(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in a:
        if k in b:
            out[k] = a[k] - b[k]
    return out

def tree_scale(a: Mapping[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    return {k: v * s for k, v in a.items()}

def tree_apply_(a: MutableMapping[str, torch.Tensor], fn: Callable[[torch.Tensor, str], torch.Tensor]) -> None:
    for k, v in a.items():
        a[k] = fn(v, k)

def tree_copy_into_(dst: MutableMapping[str, torch.Tensor], src: Mapping[str, torch.Tensor]) -> None:
    for k, v in src.items():
        if k in dst:
            dst[k].copy_(v)


# -----------------------------
# Config
# -----------------------------
@dataclass
class PUMConfig:
    m: int = 6
    alpha: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    # σ strategy: either fixed scalar on every layer or per-layer RMS * κ
    sigma_mode: str = "rms_kappa"   # {"fixed", "rms_kappa"}
    sigma_fixed: float = 0.05        # used if sigma_mode == "fixed"
    kappa: float = 0.10              # used if sigma_mode == "rms_kappa"
    eta_srv: float = 1.0
    seed_noise: int = 17
    seed_reparam: int = 23

    # Reparam toggles
    reparam_attention_rotate: bool = True
    reparam_ffn_pair_permute: bool = True
    reparam_residual_permute: bool = False  # optional; False by default for simplicity/robustness

    # Llama-style hints (only used if available from model.config; otherwise inferred/bypassed)
    attn_num_heads: int | None = None
    attn_num_kv_heads: int | None = None  # not needed for rotations but present in some configs

    # Verbose logging
    verbose: bool = True


# -----------------------------
# RNG helpers: deterministic per (seed, name)
# -----------------------------
def _seed_from(scope_seed: int, *chunks: str) -> int:
    h = hashlib.sha256()
    h.update(scope_seed.to_bytes(8, "little", signed=False))
    for c in chunks:
        h.update(c.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little")

def _torch_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# -----------------------------
# Sigma per layer
# -----------------------------
def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() * x.float())).item() + 1e-12)

def compute_sigma_by_layer(params: Mapping[str, torch.Tensor], cfg: PUMConfig) -> Dict[str, float]:
    out = {}
    if cfg.sigma_mode == "fixed":
        for k in params:
            out[k] = float(cfg.sigma_fixed)
    elif cfg.sigma_mode == "rms_kappa":
        for k, v in params.items():
            out[k] = float(cfg.kappa * _rms(v))
    else:
        raise ValueError(f"Unknown sigma_mode: {cfg.sigma_mode}")
    return out


# -----------------------------
# Noise: ε_k,ℓ = α_k ε^0_k,ℓ  (linearly dependent; not zero-sum after scaling)
# -----------------------------
def generate_ld_noise_by_copy(
    params: Mapping[str, torch.Tensor],
    sigmas: Mapping[str, float],
    cfg: PUMConfig,
) -> List[Dict[str, torch.Tensor]]:
    """
    Returns: list length m with per-copy noise state_dicts (same keys as params).
    Implementation:
      - For each key ℓ: draw z_{k,ℓ} ~ N(0, σ_ℓ^2 I), compute ε^0_{k,ℓ} = sqrt(m/(m-1)) (z_{k,ℓ} - mean_k z_{k,ℓ})
      - Publish ε_{k,ℓ} = α_k ε^0_{k,ℓ}  (linearly dependent across copies; they no longer sum to zero)
    """
    m = cfg.m
    alpha = cfg.alpha
    assert len(alpha) == m, f"len(alpha)={len(alpha)} must equal m={m}"

    # Pre-allocate
    eps0_by_key = {k: [None] * m for k in params.keys()}
    eps_by_copy = [tree_zeros_like(params, device=list(params.values())[0].device) for _ in range(m)]

    for name, w in params.items():
        sigma = sigmas[name]
        gen_key = _torch_gen(_seed_from(cfg.seed_noise, f"noise::{name}"))
        # Draw z_k
        zs = []
        for k in range(m):
            z = torch.normal(
                mean=0.0,
                std=sigma,
                size=w.shape,
                generator=gen_key,
                device=w.device,
                dtype=w.dtype,
            )
            zs.append(z)
        z_bar = sum(zs) / m
        scale = math.sqrt(m / (m - 1))
        for k in range(m):
            eps0 = scale * (zs[k] - z_bar)
            eps0_by_key[name][k] = eps0

        # scale by α_k to get published noise
        for k in range(m):
            eps_by_copy[k][name] = alpha[k] * eps0_by_key[name][k]

    return eps_by_copy


# -----------------------------
# Reparameterization planning for HF Llama-style models (robust fallback to identity)
# -----------------------------
class ReparamPlan:
    """
    Collects how to reparameterize the model:
      - attention rotations (per head): q_proj, k_proj, v_proj (left-mul), o_proj (right-mul with transpose)
      - FFN pair permutations (SwiGLU): gate_proj, up_proj (row permute), down_proj (col permute)
    If a pattern is not found, that reparam is silently skipped (identity).
    """

    def __init__(self, model: nn.Module, cfg: PUMConfig):
        self.cfg = cfg
        self.model = model
        sd = model.state_dict()
        self.keys = list(sd.keys())

        # discover layers by common HF Llama key patterns
        self.attn_blocks: List[Dict[str, str]] = []  # dict with keys: q,k,v,o
        self.ffn_blocks: List[Dict[str, str]] = []   # dict with keys: gate, up, down
        self.residual_norms: List[Tuple[str, str]] = []  # optional

        # Scan for layer indices present
        # Typical keys: model.layers.{i}.self_attn.q_proj.weight etc.
        i = 0
        while True:
            prefix = f"model.layers.{i}."
            qk = prefix + "self_attn.q_proj.weight"
            kk = prefix + "self_attn.k_proj.weight"
            vk = prefix + "self_attn.v_proj.weight"
            ok = prefix + "self_attn.o_proj.weight"
            if qk in sd and kk in sd and vk in sd and ok in sd:
                self.attn_blocks.append({"q": qk, "k": kk, "v": vk, "o": ok})
            else:
                # Stop when a first miss happens after the first non-empty
                if i > 0:
                    break

            gate = prefix + "mlp.gate_proj.weight"
            up = prefix + "mlp.up_proj.weight"
            down = prefix + "mlp.down_proj.weight"
            if gate in sd and up in sd and down in sd:
                self.ffn_blocks.append({"gate": gate, "up": up, "down": down})
            i += 1

        # Residual/norm permutations (optional)
        # Typical keys: input_layernorm.weight, post_attention_layernorm.weight
        if cfg.reparam_residual_permute:
            j = 0
            while True:
                ln1 = f"model.layers.{j}.input_layernorm.weight"
                ln2 = f"model.layers.{j}.post_attention_layernorm.weight"
                if ln1 in sd and ln2 in sd:
                    self.residual_norms.append((ln1, ln2))
                else:
                    if j > 0:
                        break
                j += 1

        # Determine head hints
        self.hidden_size = None
        self.num_heads = cfg.attn_num_heads
        if self.attn_blocks:
            wq = sd[self.attn_blocks[0]["q"]]
            self.hidden_size = wq.shape[0]  # rows = output size
            if self.num_heads is None and hasattr(getattr(model, "config", None), "num_attention_heads"):
                self.num_heads = int(getattr(model.config, "num_attention_heads"))
            # if still None, try a few divisors
            if self.num_heads is None:
                for cand in [64, 48, 40, 32, 16, 8, 4, 2]:
                    if self.hidden_size % cand == 0:
                        self.num_heads = cand
                        break

        if cfg.verbose:
            logger.info(
                "ReparamPlan: attn_blocks=%d, ffn_blocks=%d, residual_norm_pairs=%d, hidden_size=%s, num_heads=%s",
                len(self.attn_blocks), len(self.ffn_blocks), len(self.residual_norms),
                str(self.hidden_size), str(self.num_heads)
            )

    def _sample_head_rotation(self, gen: torch.Generator) -> torch.Tensor:
        """
        Returns a block-diagonal R in O(hidden_size) with one orthogonal block per head-size.
        """
        if self.hidden_size is None or self.num_heads is None or self.hidden_size % self.num_heads != 0:
            return None
        head_dim = self.hidden_size // self.num_heads
        device = torch.device("cpu")
        blocks = []
        for _ in range(self.num_heads):
            A = torch.randn((head_dim, head_dim), generator=gen, device=device)
            Q, _ = torch.linalg.qr(A, mode="reduced")
            # Ensure det(Q)=+1 (proper rotation)
            if torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            blocks.append(Q)
        R = torch.block_diag(*blocks)
        return R

    def _apply_attention_rotation_(self, sd: MutableMapping[str, torch.Tensor], R: torch.Tensor, inverse: bool = False):
        """
        W_Q' = R W_Q; W_K' = R W_K; W_V' = R W_V; W_O' = W_O R^T
        If inverse, apply the inverse transforms.
        """
        if R is None:
            return
        Rt = R.t()
        for blk in self.attn_blocks:
            WQ = sd[blk["q"]]; WK = sd[blk["k"]]; WV = sd[blk["v"]]; WO = sd[blk["o"]]
            if not inverse:
                sd[blk["q"]] = (R @ WQ)
                sd[blk["k"]] = (R @ WK)
                sd[blk["v"]] = (R @ WV)
                sd[blk["o"]] = (WO @ Rt)
            else:
                sd[blk["q"]] = (Rt @ WQ)
                sd[blk["k"]] = (Rt @ WK)
                sd[blk["v"]] = (Rt @ WV)
                sd[blk["o"]] = (WO @ R)

    def _apply_ffn_pair_permutation_(self, sd: MutableMapping[str, torch.Tensor], perm: torch.Tensor, inverse: bool = False):
        """
        gate', up' = P * (gate, up); down' = down * P^T
        Shapes: gate/up: [intermediate, hidden]; down: [hidden, intermediate]
        """
        if perm is None:
            return
        P = perm
        Pt = perm.t()
        for blk in self.ffn_blocks:
            G = sd[blk["gate"]]
            U = sd[blk["up"]]
            D = sd[blk["down"]]
            if not inverse:
                sd[blk["gate"]] = P @ G
                sd[blk["up"]]   = P @ U
                sd[blk["down"]] = D @ Pt
            else:
                sd[blk["gate"]] = Pt @ G
                sd[blk["up"]]   = Pt @ U
                sd[blk["down"]] = D @ P

    def _sample_pair_permutation(self, sd: Mapping[str, torch.Tensor], gen: torch.Generator) -> torch.Tensor:
        """
        Build a paired-channel permutation for SwiGLU blocks.
        We infer 'intermediate' dim from gate_proj rows. We permute by head-aligned pairs if possible; otherwise random.
        """
        if not self.ffn_blocks:
            return None
        any_gate = sd[self.ffn_blocks[0]["gate"]]
        inter = any_gate.shape[0]  # rows
        # enforce even for SwiGLU pairing
        if inter % 2 != 0:
            inter -= 1
        idx = torch.arange(inter)
        # pair indices as (0,1), (2,3), ...
        pairs = idx.view(-1, 2).tolist()
        # shuffle pairs deterministically
        rng = random.Random(torch.seed() if gen is None else int(torch.randint(0, 2**31-1, (1,), generator=gen)))
        rng.shuffle(pairs)
        perm_list = [i for pair in pairs for i in pair]
        # if shapes had an extra last row (odd), append identity for it
        if any_gate.shape[0] > inter:
            perm_list.append(any_gate.shape[0] - 1)
        P = torch.eye(any_gate.shape[0])
        P = P[perm_list, :]
        return P

    def sample_T(self, k: int, cfg: PUMConfig) -> Tuple[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
                                                        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]:
        """
        Returns (T_fwd, T_inv) which map entire state_dict linearly.
        """
        # Build transforms deterministically from (seed_reparam, k)
        g_attn = _torch_gen(_seed_from(cfg.seed_reparam, f"attn::{k}"))
        g_ffn = _torch_gen(_seed_from(cfg.seed_reparam, f"ffn::{k}"))

        R = self._sample_head_rotation(g_attn) if cfg.reparam_attention_rotate else None
        P = self._sample_pair_permutation(self.model.state_dict(), g_ffn) if cfg.reparam_ffn_pair_permute else None

        def T_fwd(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            sd = tree_like(sd_in)
            self._apply_attention_rotation_(sd, R, inverse=False)
            self._apply_ffn_pair_permutation_(sd, P, inverse=False)
            # residual/norm permutations can be added here if you enable and define Pi
            return sd

        def T_inv(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            sd = tree_like(sd_in)
            self._apply_ffn_pair_permutation_(sd, P, inverse=True)
            self._apply_attention_rotation_(sd, R, inverse=True)
            return sd

        return T_fwd, T_inv


# -----------------------------
# Main runner
# -----------------------------
class PUMTrainer:
    """
    Orchestrates one PUM–LD unlearning round.
    You provide:
      - model: nn.Module
      - client_unlearn_fn: Callable[[state_dict], state_dict_delta_in_same_space]
        (runs local unlearning starting at the provided reparameterized noisy copy and returns Δ)
    """

    def __init__(self, model: nn.Module, cfg: PUMConfig):
        self.model = model
        self.cfg = cfg
        if cfg.verbose:
            logger.setLevel(logging.INFO)

    @torch.no_grad()
    def run_round(
        self,
        client_unlearn_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns:
          (new_state_dict, bar_delta) where:
            new_state_dict is the updated server model θ_r,
            bar_delta is the aggregated (harmonically de-noised) update in server (original) parameter space.
        """
        cfg = self.cfg
        model = self.model
        base_sd = model.state_dict()
        device = next(model.parameters()).device

        # 1) compute σ_ℓ per layer
        sigmas = compute_sigma_by_layer(base_sd, cfg)

        # 2) generate ε_k (linearly dependent)
        eps_by_copy = generate_ld_noise_by_copy(base_sd, sigmas, cfg)

        # 3) prepare reparam plan
        plan = ReparamPlan(model, cfg)

        # 4) accumulators for harmonic de-noising
        S0 = 0.0
        S1 = tree_zeros_like(base_sd, device=device)

        # 5) iterate copies
        for k in range(cfg.m):
            T_fwd, T_inv = plan.sample_T(k, cfg)
            # publish copy
            theta_pub = T_fwd(tree_add(base_sd, eps_by_copy[k]))
            # client's local unlearning (on a copy; returns Δ in the published (T-space))
            delta_client = client_unlearn_fn(theta_pub)
            # invert back to server/original parameter space
            delta_srv_space = T_inv(delta_client)
            w = 1.0 / cfg.alpha[k]
            S0 += w
            S1 = tree_add(S1, tree_scale(delta_srv_space, w))

        # 6) aggregate / de-noise (harmonic)
        bar_delta = tree_scale(S1, 1.0 / S0)

        # 7) update server model
        new_sd = tree_add(base_sd, tree_scale(bar_delta, cfg.eta_srv))
        tree_copy_into_(base_sd, new_sd)  # in-place update of model state_dict buffers
        model.load_state_dict(base_sd)
        return new_sd, bar_delta
