# src/trainer/unlearn/pum.py
# PUM–LD: No clipping; fixed σ; α-scaled linearly dependent copy noise; T_k reparams; harmonic de-noise

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, MutableMapping, Tuple
import math
import hashlib
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


# -----------------------------
# Small tree utilities over state_dict
# -----------------------------
def tree_like(x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in x.items()}

def tree_zeros_like(x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in x.items()}

def tree_add(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] + b[k] for k in a.keys() if k in b}

def tree_sub(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] - b[k] for k in a.keys() if k in b}

def tree_scale(a: Mapping[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    return {k: v * s for k, v in a.items()}

def tree_copy_into_(dst: MutableMapping[str, torch.Tensor], src: Mapping[str, torch.Tensor]) -> None:
    for k, v in src.items():
        if k in dst:
            dst[k].copy_(v)


# -----------------------------
# Config
# -----------------------------
@dataclass
class PUMConfig:
    # number of copies
    m: int
    # α scaling per copy (must match m)
    alpha: List[float]

    # σ strategy: fixed scalar or per-layer RMS * κ
    sigma_mode: str = "rms_kappa"   # {"fixed", "rms_kappa"}
    sigma_fixed: float
    kappa: float

    # server step size on aggregated delta
    eta_srv: float = 1.0

    # deterministic seeds
    seed_noise: int = 0
    seed_reparam: int = 1

    # Reparam toggles
    reparam_attention_rotate: bool = True
    reparam_ffn_pair_permute: bool = True
    reparam_residual_permute: bool = False  # optional

    # Hint in case model.config lacks this
    attn_num_heads: int | None = None

    verbose: bool = True


# -----------------------------
# RNG helpers
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
# σ per layer
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
    For each key ℓ:
      - draw z_{k,ℓ} ~ N(0, σ_ℓ^2 I)
      - ε^0_{k,ℓ} = sqrt(m/(m-1)) (z_{k,ℓ} - mean_k z_{k,ℓ})
      - publish ε_{k,ℓ} = α_k ε^0_{k,ℓ}  (linearly dependent across copies; not zero-sum)
    """
    m = cfg.m
    alpha = cfg.alpha
    assert len(alpha) == m, f"len(alpha)={len(alpha)} must equal m={m}"

    eps0_by_key: Dict[str, List[torch.Tensor]] = {k: [None] * m for k in params.keys()}
    eps_by_copy: List[Dict[str, torch.Tensor]] = [tree_zeros_like(params) for _ in range(m)]

    for name, w in params.items():
        sigma = sigmas[name]
        gen_key = _torch_gen(_seed_from(cfg.seed_noise, f"noise::{name}"))
        # z_k ~ N(0, σ^2 I)
        zs = [torch.normal(0.0, sigma, size=w.shape, generator=gen_key, device=w.device, dtype=w.dtype) for _ in range(m)]
        z_bar = sum(zs) / m
        scale = math.sqrt(m / (m - 1.0))
        for k in range(m):
            eps0_by_key[name][k] = scale * (zs[k] - z_bar)
        # publish ε_k = α_k ε^0_k
        for k in range(m):
            eps_by_copy[k][name] = cfg.alpha[k] * eps0_by_key[name][k]

    return eps_by_copy

##-----OK checked

# -----------------------------
# Reparameterization planner (HF Llama-style keys with robust fallbacks)
# -----------------------------
class ReparamPlan:
    """
    attention rotations (per head):
        W_Q' = R W_Q; W_K' = R W_K; W_V' = R W_V; W_O' = W_O R^T
    FFN paired permutations (SwiGLU):
        gate', up' = P (gate, up); down' = down P^T
    All transforms are linear & invertible; exact inverses applied on server.
    """

    def __init__(self, model: nn.Module, cfg: PUMConfig):
        self.cfg = cfg
        self.model = model
        sd = model.state_dict()
        self.attn_blocks: List[Dict[str, str]] = []
        self.ffn_blocks: List[Dict[str, str]] = []

        # Discover layer blocks by common HF Llama naming
        i = 0
        while True:
            prefix = f"model.layers.{i}."
            qk = prefix + "self_attn.q_proj.weight"
            kk = prefix + "self_attn.k_proj.weight"
            vk = prefix + "self_attn.v_proj.weight"
            ok = prefix + "self_attn.o_proj.weight"
            gate = prefix + "mlp.gate_proj.weight"
            up = prefix + "mlp.up_proj.weight"
            down = prefix + "mlp.down_proj.weight"

            found_attn = (qk in sd and kk in sd and vk in sd and ok in sd)
            found_ffn = (gate in sd and up in sd and down in sd)
            if found_attn:
                self.attn_blocks.append({"q": qk, "k": kk, "v": vk, "o": ok})

            if found_ffn:
                self.ffn_blocks.append({"gate": gate, "up": up, "down": down})

            if not found_attn and not found_ffn:
                if i > 0:  # stop after first miss following at least one hit
                    break
            i += 1

#-----prompt: may add check warning if nothing is found
        # head hints
        self.hidden_size = None
        self.num_heads = cfg.attn_num_heads
        if self.attn_blocks:
            wq = sd[self.attn_blocks[0]["q"]]
            self.hidden_size = wq.shape[0]
            if self.num_heads is None and hasattr(getattr(model, "config", None), "num_attention_heads"):
                self.num_heads = int(getattr(model.config, "num_attention_heads"))
            if self.num_heads is None:
                for cand in [64, 48, 40, 32, 16, 8, 4, 2]:
                    if self.hidden_size % cand == 0:
                        self.num_heads = cand
                        break

        if cfg.verbose:
            logger.info(
                "ReparamPlan: attn=%d, ffn=%d, hidden=%s, n_heads=%s",
                len(self.attn_blocks), len(self.ffn_blocks),
                str(self.hidden_size), str(self.num_heads)
            )

    def _sample_head_rotation(self, gen: torch.Generator, device, dtype) -> torch.Tensor | None:
        if self.hidden_size is None or self.num_heads is None or self.hidden_size % self.num_heads != 0:
            return None
        head_dim = self.hidden_size // self.num_heads
        blocks = []
        for _ in range(self.num_heads):
            A = torch.randn((head_dim, head_dim), generator=gen, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(A, mode="reduced")
            # proper rotation
            if torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            blocks.append(Q)
        R = torch.block_diag(*blocks)
        return R

    def _apply_attention_rotation_(self, sd: MutableMapping[str, torch.Tensor], R: torch.Tensor, inverse: bool = False):
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

    def _apply_ffn_pair_permutation_(self, sd: MutableMapping[str, torch.Tensor], P: torch.Tensor, inverse: bool = False):
        if P is None:
            return
        Pt = P.t()
        for blk in self.ffn_blocks:
            G = sd[blk["gate"]]; U = sd[blk["up"]]; D = sd[blk["down"]]
            if not inverse:
                sd[blk["gate"]] = P @ G
                sd[blk["up"]]   = P @ U
                sd[blk["down"]] = D @ Pt
            else:
                sd[blk["gate"]] = Pt @ G
                sd[blk["up"]]   = Pt @ U
                sd[blk["down"]] = D @ P

    def _sample_pair_permutation(self, sd: Mapping[str, torch.Tensor], gen: torch.Generator, device, dtype) -> torch.Tensor | None:
        if not self.ffn_blocks:
            return None
        any_gate = sd[self.ffn_blocks[0]["gate"]]
        inter = any_gate.shape[0]
        even = inter - (inter % 2)
        # permute pairs (0,1), (2,3), ...
        pair_idx = torch.arange(even, device=device)
        pair_idx = pair_idx.view(-1, 2)
        # deterministic shuffle via generator
        perm_pairs = pair_idx[torch.randperm(pair_idx.size(0), generator=gen, device=device)]
        perm_list = perm_pairs.reshape(-1).tolist()
        if inter > even:
            perm_list.append(inter - 1)
        P = torch.eye(inter, device=device, dtype=dtype)[perm_list, :]
        return P

    def sample_T(self, k: int, cfg: PUMConfig) -> Tuple[
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ]:
        g_attn = _torch_gen(_seed_from(cfg.seed_reparam, f"attn::{k}"))
        g_ffn  = _torch_gen(_seed_from(cfg.seed_reparam, f"ffn::{k}"))
        # device/dtype choices: follow first param
        sd = self.model.state_dict()
        any_key, any_w = next(iter(sd.items()))
        device, dtype = any_w.device, any_w.dtype

        R = self._sample_head_rotation(g_attn, device, dtype) if cfg.reparam_attention_rotate else None
        P = self._sample_pair_permutation(sd, g_ffn, device, dtype) if cfg.reparam_ffn_pair_permute else None

        def T_fwd(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            sd_copy = tree_like(sd_in)
            self._apply_attention_rotation_(sd_copy, R, inverse=False)
            self._apply_ffn_pair_permutation_(sd_copy, P, inverse=False)
            return sd_copy

        def T_inv(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            sd_copy = tree_like(sd_in)
            self._apply_ffn_pair_permutation_(sd_copy, P, inverse=True)
            self._apply_attention_rotation_(sd_copy, R, inverse=True)
            return sd_copy

        return T_fwd, T_inv


# -----------------------------
# Main trainer
# -----------------------------
class PUMTrainer:
    """
    Orchestrates one PUM–LD unlearning round.
    You pass in a `client_unlearn_fn(state_dict) -> delta_state_dict` that
    runs local unlearning from the given (reparameterized + noisy) copy and
    returns the parameter delta measured in the same space as the input.
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
        cfg = self.cfg
        base_sd = self.model.state_dict()

        # σ per layer (fixed strategy)
        sigmas = compute_sigma_by_layer(base_sd, cfg)

        # ε_k (linearly dependent)
        eps_by_copy = generate_ld_noise_by_copy(base_sd, sigmas, cfg)

        # reparam plan
        plan = ReparamPlan(self.model, cfg)

        # harmonic accumulators
        S0 = 0.0
        S1 = tree_zeros_like(base_sd)

        for k in range(cfg.m):
            T_fwd, T_inv = plan.sample_T(k, cfg)
            theta_pub = T_fwd(tree_add(base_sd, eps_by_copy[k]))
            delta_client = client_unlearn_fn(theta_pub)     # Δ in T-space
            delta_srv = T_inv(delta_client)                 # back to server/original
            w = 1.0 / cfg.alpha[k]
            S0 += w
            S1 = tree_add(S1, tree_scale(delta_srv, w))

        bar_delta = tree_scale(S1, 1.0 / S0)
        new_sd = tree_add(base_sd, tree_scale(bar_delta, cfg.eta_srv))

        # load updated weights
        base_sd.update(new_sd)
        self.model.load_state_dict(base_sd)
        return new_sd, bar_delta



"""

sample run

# fresh env, per README (nvcc + flash-attn as needed)
conda activate unlearning
pip install -e ".[lm_eval]"
python setup_data.py --eval

# then:
bash scripts/pum_unlearn.sh"""