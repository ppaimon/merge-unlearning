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

import math
logger = logging.getLogger(__name__)


# -----------------------------
# Small tree utilities over state_dict
# -----------------------------


# clone one mapping into Dict
def tree_like(x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in x.items()}

# Build dict with zero valued (Tensor) Dict
def tree_zeros_like(x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in x.items()}

# add the values with same key, return Dict
def tree_add(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k].to('cpu') + b[k].to('cpu') for k in a.keys() if k in b}

# subtract values with same key, return Dict
def tree_sub(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k].to('cpu') - b[k].to('cpu') for k in a.keys() if k in b}

# Multiply the values with scaling s
def tree_scale(a: Mapping[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    return {k: v * s for k, v in a.items()}

# copy common items from second var to the first
def tree_copy_into_(dst: MutableMapping[str, torch.Tensor], src: Mapping[str, torch.Tensor]) -> None:
    for k, v in src.items():
        if k in dst:
            dst[k].copy_(v)


# -----------------------------
# Config
# -----------------------------

from dataclasses import dataclass
from typing import List, Dict, MutableMapping, Mapping, Tuple, Optional

@dataclass
class PUMConfig:
    # --------- non-defaults (must come first) ---------
    m: int
    alpha: List[float]
    sigma_mode: str              # {"fixed", "rms_kappa"}
    kappa: float
    seed_noise: int
    seed_reparam: int
    seed_train: int

    # --------- defaults ---------
    sigma_fixed: float = 0.0
    eta_srv: float = 1.0

    # Reparam toggles ...
    reparam_attention_rotate: bool = True
    reparam_attention_rope_aware: bool = True
    reparam_ffn_pair_permute: bool = True
    reparam_residual_permute: bool = False

    attn_num_heads: Optional[int] = None
    attn_num_kv_heads: Optional[int] = None

    verbose: bool = True

    # --- NEW ---
    sigma_ref: str = "params"          # {"params","task_vector"}；"task_vector" 时按任务向量计算 σ
    noise_generator: str = "gaussian"  # {"gaussian","uni","cos"}



# -----------------------------
# RNG helpers
# -----------------------------
def _seed_from(scope_seed: int, *chunks: str) -> int:
    h = hashlib.sha256()
    h.update(scope_seed.to_bytes(8, "little", signed=False))
    for c in chunks:
        h.update(c.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little")

def _torch_gen(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


# -----------------------------
# σ per layer
# -----------------------------
def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() * x.float())).item() + 1e-12)

def compute_sigma_by_layer(
    params: Mapping[str, torch.Tensor],
    cfg: PUMConfig,
    ref_map: Mapping[str, torch.Tensor] | None = None,   # NEW
) -> Dict[str, float]:
    """
    当 cfg.sigma_mode == "rms_kappa" 时：
      - 若 cfg.sigma_ref == "task_vector"，则从 ref_map[k] 取参考张量（须与 params 对齐），
        每层 σ_ℓ = κ * RMS(ref_map[ℓ])；
      - 否则，按当前参数 params[ℓ] 求 RMS（原有行为）。
    """
    out: Dict[str, float] = {}
    if cfg.sigma_mode == "fixed":
        for k in params:
            out[k] = float(cfg.sigma_fixed)
        return out

    if cfg.sigma_mode == "rms_kappa":
        use_task_vec = (cfg.sigma_ref.lower() == "task_vector")
        if use_task_vec and ref_map is None:
            raise ValueError("sigma_ref='task_vector' 需要提供 ref_map（例如任务向量）。")

        for k, v in params.items():
            base_t = ref_map[k] if (use_task_vec and (k in ref_map)) else v
            out[k] = float(cfg.kappa * _rms(base_t))
        return out

    raise ValueError(f"Unknown sigma_mode: {cfg.sigma_mode}")


# -----------------------------
# Noise: ε_k,ℓ = α_k ε^0_k,ℓ  (linearly dependent; not zero-sum after scaling)
# return eps_by_copy[k][name] , each is a noise tensor, with k be the copy index and name be the layername
# -----------------------------
def generate_ld_noise_by_copy(
    params: Mapping[str, torch.Tensor],
    sigmas: Mapping[str, float],
    cfg: PUMConfig,
) -> List[Dict[str, torch.Tensor]]:
    """
    ε^0_{k,ℓ} = sqrt(m/(m-1)) (z_{k,ℓ} - mean_k z_{k,ℓ}),  ε_{k,ℓ} = α_k ε^0_{k,ℓ}
    其中 z_{k,ℓ} 的元素独立同分布，分布由 cfg.noise_generator 控制：
      - 'gaussian' : N(0, σ_ℓ^2)
      - 'uni'      : U[-σ_ℓ, σ_ℓ]
      - 'cos'      : σ_ℓ * cos(π * U[0,1])
    """
    m = cfg.m
    assert len(cfg.alpha) == m, f"len(alpha)={len(cfg.alpha)} must equal m={m}"
    mode = cfg.noise_generator.lower()
    eps_by_copy: List[Dict[str, torch.Tensor]] = [tree_zeros_like(params) for _ in range(m)]

    def _draw(shape, sigma, gen, dtype) -> torch.Tensor:
        if sigma == 0.0:
            return torch.zeros(shape, device="cpu", dtype=torch.float32)
        if mode in ("gaussian", "normal"):
            z = torch.normal(0.0, sigma, size=shape, generator=gen, device="cpu", dtype=torch.float32)
        elif mode in ("uni", "uniform"):
            u = torch.rand(shape, generator=gen, device="cpu", dtype=torch.float32)
            z = (2.0 * u - 1.0) * sigma
        elif mode in ("cos", "cosine"):
            u = torch.rand(shape, generator=gen, device="cpu", dtype=torch.float32)
            z = sigma * torch.cos(math.pi * u)
        else:
            raise ValueError(f"Unknown noise_generator: {cfg.noise_generator}")
        return z.to(dtype=dtype)

    for name, w in params.items():
        sigma = float(sigmas[name])
        gen_key = _torch_gen(_seed_from(cfg.seed_noise, f"noise::{name}::{mode}"), device="cpu")

        # i.i.d. draws per-copy
        zs = [_draw(w.shape, sigma, gen_key, dtype=torch.float32) for _ in range(m)]
        z_bar = sum(zs) / m
        scale = math.sqrt(m / (m - 1.0))
        eps0 = [scale * (zk - z_bar) for zk in zs]  # zero-sum across copies

        for k in range(m):
            # 发布噪声：ε_k = α_k ε^0_k
            eps_by_copy[k][name] = (cfg.alpha[k] * eps0[k]).to(dtype=w.dtype)

    return eps_by_copy


##-----OK checked

# -----------------------------
# Reparameterization planner (HF Llama-style keys with robust fallbacks)
# -----------------------------
class ReparamPlan:
    """
    Attention reparameterization for general head layouts (MHA/GQA/MQA), with
    orthogonal KV blocks S_KV and their lift U(A,S_KV) (no head permutations):

        W_Q' = W_Q U(A,S_KV)
        W_K' = W_K S_KV
        W_V' = W_V S_KV
        W_O' = U(A,S_KV)^T W_O

    The inverse uses transposes only. FFN pair permutations are unchanged.
    """

    def __init__(self, model: nn.Module, cfg: PUMConfig):
        self.cfg = cfg
        self.model = model
        sd = model.state_dict()

        # discover layer blocks (same as before)
        self.attn_blocks: List[Dict[str, str]] = []
        self.ffn_blocks: List[Dict[str, str]] = []
        i = 0
        while True:
            prefix = f"model.layers.{i}."

            # attention weights
            qk = prefix + "self_attn.q_proj.weight"
            kk = prefix + "self_attn.k_proj.weight"
            vk = prefix + "self_attn.v_proj.weight"
            ok = prefix + "self_attn.o_proj.weight"

            # optional attention biases
            qb = prefix + "self_attn.q_proj.bias"
            kb = prefix + "self_attn.k_proj.bias"
            vb = prefix + "self_attn.v_proj.bias"
            ob = prefix + "self_attn.o_proj.bias"

            # ffn weights
            gate = prefix + "mlp.gate_proj.weight"
            up   = prefix + "mlp.up_proj.weight"
            down = prefix + "mlp.down_proj.weight"

            # optional ffn biases
            gate_b = prefix + "mlp.gate_proj.bias"
            up_b   = prefix + "mlp.up_proj.bias"
            down_b = prefix + "mlp.down_proj.bias"

            found_attn = (qk in sd and kk in sd and vk in sd and ok in sd)
            found_ffn  = (gate in sd and up in sd and down in sd)

            if found_attn:
                blk = {"q": qk, "k": kk, "v": vk, "o": ok}
                if qb in sd: blk["qb"] = qb
                if kb in sd: blk["kb"] = kb
                if vb in sd: blk["vb"] = vb
                if ob in sd: blk["ob"] = ob  # (bias of o_proj stays unchanged in our map)
                self.attn_blocks.append(blk)

            if found_ffn:
                blk = {"gate": gate, "up": up, "down": down}
                if gate_b in sd: blk["gate_b"] = gate_b
                if up_b in sd:   blk["up_b"]   = up_b
                if down_b in sd: blk["down_b"] = down_b
                self.ffn_blocks.append(blk)

            if (not found_attn) and (not found_ffn):
                if i > 0: break
                else: raise ValueError("no layer is found for model")
            i += 1


        self.H_Q  = cfg.attn_num_heads or int(getattr(getattr(model, "config", None), "num_attention_heads", 0)) or None
        self.H_KV = cfg.attn_num_kv_heads or int(getattr(getattr(model, "config", None), "num_key_value_heads", 0)) or None
        if self.H_Q is None:
            wq = sd[self.attn_blocks[0]["q"]]  # [out, in]
            # fall back: try common divisors of the *row* dim
            row = wq.shape[0]
            for cand in [64, 48, 40, 32, 24, 16, 12, 8, 6, 4, 2]:
                if row % cand == 0: self.H_Q = cand; break
        if self.H_KV is None:
            self.H_KV = self.H_Q  # default to MHA

        wq = sd[self.attn_blocks[0]["q"]]  # [out, in] = [(H_Q*d_h), d_model]
        self.row_dim = wq.shape[0]
        self.in_dim  = wq.shape[1]
        assert self.row_dim % self.H_Q == 0, f"Output dim {self.row_dim} not divisible by H_Q={self.H_Q}"
        self.d_h = self.row_dim // self.H_Q
        assert (self.H_Q % self.H_KV) == 0, f"H_Q={self.H_Q} must be a multiple of H_KV={self.H_KV} (no head perms)."
        self.g = self.H_Q // self.H_KV  # queries per KV group

        if cfg.verbose:
            logger.info(
                "ReparamPlan[MHA/GQA]: H_Q=%s, H_KV=%s, d_h=%s, row_dim=%s, in_dim=%s, layers=%d",
                self.H_Q, self.H_KV, self.d_h, self.row_dim, self.in_dim, len(self.attn_blocks)
            )


    def _rope_block_rotation(self, gen: torch.Generator, device, dtype) -> torch.Tensor:
        """
        LLaMA-style RoPE pairs channels as (i, i + d_h//2) (half-split), not (2i, 2i+1).
        Build S that is a product of independent 2×2 rotations on those pairs:

            S = [[diag(cos φ), -diag(sin φ)],
                [diag(sin φ),  diag(cos φ)]]

        This S commutes with HF's apply_rotary_pos_emb implementation.
        """
        dh = self.d_h
        assert dh % 2 == 0, "RoPE-aware rotation requires even head_dim for LLaMA-style pairing"
        n = dh // 2

        angles = 2.0 * math.pi * torch.rand(n, generator=gen, device=device, dtype=torch.float32)
        c = torch.cos(angles).to(dtype=dtype)
        s = torch.sin(angles).to(dtype=dtype)

        Dc = torch.diag(c).to(device=device, dtype=dtype)
        Ds = torch.diag(s).to(device=device, dtype=dtype)

        top = torch.cat([Dc, -Ds], dim=1)   # (n, 2n)
        bot = torch.cat([Ds,  Dc], dim=1)   # (n, 2n)
        S = torch.cat([top, bot], dim=0)    # (2n, 2n) == (dh, dh)
        return S


    def _qr_orthogonal(self, gen: torch.Generator, device, dtype) -> torch.Tensor:
        A = torch.randn((self.d_h, self.d_h), generator=gen, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q.to(dtype=dtype)

    def _sample_S_KV(self, gen: torch.Generator, device, dtype) -> torch.Tensor:
        """
        S_KV = blkdiag(S_1,...,S_{H_KV}).
        - If rope_aware: S_j are 2x2 rotations per RoPE plane (SO(2)^(d_h/2)).
        - Else:          S_j are QR-orthogonal in O(d_h).
        """
        blocks = []
        for _ in range(self.H_KV):
            if self.cfg.reparam_attention_rope_aware:
                S = self._rope_block_rotation(gen, device, dtype)
            else:
                S = self._qr_orthogonal(gen, device, dtype)
            blocks.append(S)
        return torch.block_diag(*blocks)


    # --- lift S_KV to queries via fixed assignment A (no permutations) ---
    def _default_assignment(self) -> List[int]:
        """π(i) = floor(i/g) with g = H_Q / H_KV; returns list of length H_Q."""
        return [i // self.g for i in range(self.H_Q)]

    def _lift_U(self, S_KV: torch.Tensor, A: List[int]) -> torch.Tensor:
        """Build U(A,S_KV) by repeating the S_p blocks on the Q-head axis."""
        # extract diagonal blocks of S_KV
        blocks = [S_KV[p*self.d_h:(p+1)*self.d_h, p*self.d_h:(p+1)*self.d_h] for p in range(self.H_KV)]
        rep = [blocks[pi] for pi in A]  # length H_Q
        return torch.block_diag(*rep)   # [H_Q*d_h, H_Q*d_h] == [col_dim, col_dim]

    def _apply_attention_T_(self, sd: MutableMapping[str, torch.Tensor], U: torch.Tensor, S_KV: torch.Tensor, inverse: bool = False):
        """
        PyTorch weight orientation:
        W_Q, W_K, W_V: (out, in)  => change of basis on 'out' via LEFT multiply
        W_O          : (out, in)  => change of basis on 'in'  via RIGHT multiply

        Forward map (matches LaTeX after transposition):
        W_Q <- U^T @ W_Q          ;  b_Q <- U^T @ b_Q (if exists)
        W_K <- S_KV^T @ W_K       ;  b_K <- S_KV^T @ b_K
        W_V <- S_KV^T @ W_V       ;  b_V <- S_KV^T @ b_V
        W_O <- W_O @ U            ;  b_O unchanged

        Inverse:
        W_Q <- U @ W_Q            ;  b_Q <- U @ b_Q
        W_K <- S_KV @ W_K         ;  b_K <- S_KV @ b_K
        W_V <- S_KV @ W_V         ;  b_V <- S_KV @ b_V
        W_O <- W_O @ U^T          ;  b_O unchanged
        """
        Ut = U.t(); SKVt = S_KV.t()
        for blk in self.attn_blocks:
            WQ = sd[blk["q"]].to('cpu')
            WK = sd[blk["k"]].to('cpu')
            WV = sd[blk["v"]].to('cpu')
            WO = sd[blk["o"]].to('cpu')
            qb = sd[blk["qb"]].to('cpu') if "qb" in blk else None
            kb = sd[blk["kb"]].to('cpu') if "kb" in blk else None
            vb = sd[blk["vb"]].to('cpu') if "vb" in blk else None
            # ob (bias of o_proj) remains unchanged

            if not inverse:
                sd[blk["q"]] = Ut @ WQ
                sd[blk["k"]] = SKVt @ WK
                sd[blk["v"]] = SKVt @ WV
                sd[blk["o"]] = WO @ U
                if qb is not None: sd[blk["qb"]] = Ut @ qb
                if kb is not None: sd[blk["kb"]] = SKVt @ kb
                if vb is not None: sd[blk["vb"]] = SKVt @ vb
            else:
                sd[blk["q"]] = U @ WQ
                sd[blk["k"]] = S_KV @ WK
                sd[blk["v"]] = S_KV @ WV
                sd[blk["o"]] = WO @ Ut
                if qb is not None: sd[blk["qb"]] = U @ qb
                if kb is not None: sd[blk["kb"]] = S_KV @ kb
                if vb is not None: sd[blk["vb"]] = S_KV @ vb


    def _apply_ffn_pair_permutation_(self, sd: MutableMapping[str, torch.Tensor], P: torch.Tensor, inverse: bool = False):
        if P is None: return
        Pt = P.t()
        for blk in self.ffn_blocks:
            G  = sd[blk["gate"]].to('cpu')
            U_ = sd[blk["up"]].to('cpu')
            D  = sd[blk["down"]].to('cpu')
            Gb = sd[blk["gate_b"]].to('cpu') if "gate_b" in blk else None
            Ub = sd[blk["up_b"]].to('cpu')   if "up_b"   in blk else None
            Db = sd[blk["down_b"]].to('cpu') if "down_b" in blk else None

            if not inverse:
                sd[blk["gate"]] = P  @ G
                sd[blk["up"]]   = P  @ U_
                sd[blk["down"]] = D  @ Pt
                if Gb is not None: sd[blk["gate_b"]] = P  @ Gb
                if Ub is not None: sd[blk["up_b"]]   = P  @ Ub
                # b2 (down bias) is on the *output* axis of W2 (unchanged by right-mul on input); keep it
            else:
                sd[blk["gate"]] = Pt @ G
                sd[blk["up"]]   = Pt @ U_
                sd[blk["down"]] = D  @ P
                if Gb is not None: sd[blk["gate_b"]] = Pt @ Gb
                if Ub is not None: sd[blk["up_b"]]   = Pt @ Ub
                # down_b unchanged in either direction



    def _sample_pair_permutation(self, sd: Mapping[str, torch.Tensor], gen: torch.Generator, device, dtype) -> torch.Tensor | None:
        # (unchanged)
        if not self.ffn_blocks: return None
        any_gate = sd[self.ffn_blocks[0]["gate"]]
        inter = any_gate.shape[0]
        even = inter - (inter % 2)
        pair_idx = torch.arange(even, device=device).view(-1, 2)
        perm_pairs = pair_idx[torch.randperm(pair_idx.size(0), generator=gen, device=device)]
        perm_list = perm_pairs.reshape(-1).tolist()
        if inter > even: perm_list.append(inter - 1)
        P = torch.eye(inter, device=device, dtype=dtype)[perm_list, :]
        return P

    def sample_T(self, k: int, cfg: PUMConfig, round_idx: Optional[int] = None):
        """
        Sample the per-copy reparameterization (attention + ffn), deterministically
        from (seed_reparam, round_idx, k).
        """
        sd = self.model.state_dict()
        any_key, any_w = next(iter(sd.items()))
        device, dtype = 'cpu', any_w.dtype

        rtag = f"r{round_idx}" if round_idx is not None else "r*"
        g_attn = _torch_gen(_seed_from(cfg.seed_reparam, f"attn::{rtag}::{k}"), device=device)
        g_ffn  = _torch_gen(_seed_from(cfg.seed_reparam, f"ffn::{rtag}::{k}"),  device=device)

        S_KV = self._sample_S_KV(g_attn, device, dtype) if cfg.reparam_attention_rotate else None
        A = self._default_assignment() if S_KV is not None else None
        U = self._lift_U(S_KV, A) if S_KV is not None else None

        P = self._sample_pair_permutation(sd, g_ffn, device, dtype) if cfg.reparam_ffn_pair_permute else None

        def T_fwd(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            sd_copy = tree_like(sd_in)
            if S_KV is not None: self._apply_attention_T_(sd_copy, U, S_KV, inverse=False)
            if P    is not None: self._apply_ffn_pair_permutation_(sd_copy, P, inverse=False)
            return sd_copy

        def T_inv(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            sd_copy = tree_like(sd_in)
            if P    is not None: self._apply_ffn_pair_permutation_(sd_copy, P, inverse=True)
            if S_KV is not None: self._apply_attention_T_(sd_copy, U, S_KV, inverse=True)
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

    def __init__(self, model: nn.Module, cfg: PUMConfig,*,
        base_ref_sd: Mapping[str, torch.Tensor] | None = None,   #
        task_vector_sd: Mapping[str, torch.Tensor] | None = None): # NEW)
        self.model = model
        self.cfg = cfg
        if cfg.verbose:
            logger.setLevel(logging.INFO)
         # --- NEW:
        self._start_sd = tree_like(model.state_dict())  #
        if task_vector_sd is not None:
            self._task_vec = task_vector_sd
        elif base_ref_sd is not None:
            # ：start_sd - base_llama3_sd
            self._task_vec = tree_sub(self._start_sd, base_ref_sd)
        else:
            self._task_vec = None  # if cfg.sigma_ref="params" 


    def run_round(
        self,
        client_unlearn_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
        round_idx: int | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        cfg = self.cfg
        base_sd = self.model.state_dict()

        # --- NEW:
        if cfg.sigma_mode == "rms_kappa" and cfg.sigma_ref.lower() == "task_vector":
            if self._task_vec is None:
                raise ValueError("cfg.sigma_ref='task_vector', but no base_ref_sd or task_vector_sd。")
            sigmas = compute_sigma_by_layer(base_sd, cfg, ref_map=self._task_vec)
        else:
            sigmas = compute_sigma_by_layer(base_sd, cfg)

        # ε_k (linearly dependent)
        eps_by_copy = generate_ld_noise_by_copy(base_sd, sigmas, cfg)

        # reparam plan
        plan = ReparamPlan(self.model, cfg)

        # harmonic accumulators
        S0 = 0.0
        S1 = tree_zeros_like(base_sd)

        for k in range(cfg.m):
            # pass round_idx through to seed the reparam deterministically for this round
            T_fwd, T_inv = plan.sample_T(k, cfg, round_idx=round_idx)
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