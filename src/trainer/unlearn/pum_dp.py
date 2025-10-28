# /home/ecs-user/PUM/src/trainer/unlearn/pum.py — Perturb–Unlearn–Merge (PUM) meta-trainer
# Public-only C_ℓ → Δ̄_{2,ℓ} → DP σ; correlated α·ε_k perturb → inner unlearning → server average.
# Deterministic seeds, tiny jitter, optional T (orthogonal + permute) reparam, center clipping with EMA ref.
# Multi-GPU safe diffs under DeepSpeed/Accelerate (gather full, unsharded state before diff).
#
# NEW (bounded calibration & multi‑round fixes):
#   • synth_mode="gaussian" (default): bounded synthetic public fallback uses per‑layer RMS, not L2.
#         C_ℓ ≈ κ · ρ · RMS(θ_ref,ℓ)  (ρ≪1, default 0.02; θ_ref is the round‑(j−1) reference)
#     This guarantees nonzero but not huge σ/C, avoiding the large values seen with function‑preserving transforms.
#   • C_ℓ monotone non‑increasing across rounds (drift fix): on round j≥2, C_ℓ^{new} ← min(C_ℓ^{old}, C_ℓ^{obs}).
#   • Relative caps/floors use per‑layer RMS(θ_ref,ℓ) (not global RMS, not L2).
#   • σ safety clips (min/max) are computed against RMS(θ_ref), not the initial base model.
#   • RDP λ grid broadened: [1.1,1.25,1.5,2,3,4,6,8,16,32,64,128,256].
#   • Critical multi‑round change: wherever “θ_base,ℓ” was implicitly used to size C_ℓ,
#     we now use the last‑round DP‑safe reference θ^{(j−1)}_ℓ (EMA of published means).
#
# Env toggles (can also be set via Hydra method_args):
#   PUM_SYNTH_MODE            : "gaussian" | "function_preserving"    (default "gaussian")
#   PUM_SYNTH_GAUSS_RHO       : float, default 0.02
#   PUM_C_REL_CLIP            : float, optional (e.g., 0.05 → 5% of RMS(layer))
#   PUM_C_REL_FLOOR           : float, optional (e.g., 0.005 → 0.5% of RMS(layer))
#   PUM_SIGMA_MIN_REL         : float, default 0.0 (min σ ≥ this · RMS(θ_ref or base))
#   PUM_SIGMA_REL_CLIP        : float, default 0.25 (max σ ≤ this · RMS(θ_ref or base))
#   PUM_SIGMA_WARMUP_FACTOR   : float, default 1.0 (e.g., 0.2 on round 1)
#
# All other mechanics stay faithful to the methodology (per-layer RDP calibration, α‑scaling, zero‑sum noise, EMA, etc.)

import os
import math
import copy
import importlib
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import TrainingArguments

from trainer.base import FinetuneTrainer

logger = logging.getLogger(__name__)

# Quick-test env caps
PUM_FORCE_MAX_ROUNDS = int(os.environ.get("PUM_FORCE_MAX_ROUNDS", "0") or "0")
PUM_FORCE_MAX_COPIES = int(os.environ.get("PUM_FORCE_MAX_COPIES", "0") or "0")
_tmp = os.environ.get("PUM_FORCE_LOCAL_MAX_STEPS", "").strip()
PUM_FORCE_LOCAL_MAX_STEPS = int(_tmp) if _tmp.isdigit() and int(_tmp) > 0 else None


# ----------------------------
# Utilities
# ----------------------------

def _near_identity_orthogonal(dim: int, eps: float, device, dtype):
    if eps <= 0:
        return torch.eye(dim, device=device, dtype=dtype)
    A = torch.randn(dim, dim, device="cpu", dtype=torch.float32)
    S = A - A.T
    M = eps * S
    R = torch.linalg.matrix_exp(M)
    return R.to(device=device, dtype=dtype)

def _canon_name(n: str) -> str:
    if n.startswith("base_model."):
        return n[len("base_model."):]
    return n

def _all_named_params(model: nn.Module):
    return [(n, p) for (n, p) in model.named_parameters() if p is not None]

def _zero_like_param_dict(model: nn.Module, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    d = {}
    for n, p in _all_named_params(model):
        d[_canon_name(n)] = torch.zeros_like(p, device=device or p.device)
    return d

def _apply_state_dict_delta(model: nn.Module, delta: Dict[str, torch.Tensor], scale: float = 1.0):
    with torch.no_grad():
        for n, p in _all_named_params(model):
            cn = _canon_name(n)
            if cn in delta:
                p.add_(scale * delta[cn].to(p.device, p.dtype))

def _diff_state_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, va in a.items():
        vb = b.get(k)
        if vb is None or va.shape != vb.shape:
            continue
        try:
            out[k] = va - vb
        except Exception:
            continue
    return out

def _randn_like_with_generator(t: torch.Tensor, std: float, gen: Optional[torch.Generator]) -> torch.Tensor:
    if std <= 0:
        return torch.zeros_like(t)
    if gen is None:
        return torch.randn_like(t) * std
    z = torch.randn(t.shape, dtype=t.dtype, device="cpu", generator=gen) * std
    return z.to(device=t.device, dtype=t.dtype)

def _generate_zero_sum_noises(
    model: nn.Module,
    m: int,
    sigma_scalar: float,
    device: Optional[torch.device] = None,
    per_param_sigma: Optional[Dict[str, float]] = None,
    rng: Optional[torch.Generator] = None,
) -> List[Dict[str, torch.Tensor]]:
    """m correlated noises with zero mean per parameter."""
    if m <= 1:
        return [_zero_like_param_dict(model, device=device)]
    if per_param_sigma is None and not (sigma_scalar and sigma_scalar > 0.0):
        return [_zero_like_param_dict(model, device=device) for _ in range(m)]

    def _std_for(n: str) -> float:
        if per_param_sigma is not None:
            s = float(per_param_sigma.get(n, 0.0))
            return s if s > 0 else 0.0
        return float(sigma_scalar)

    any_pos = False
    names = []
    for n, _ in _all_named_params(model):
        cn = _canon_name(n); names.append(cn)
        if _std_for(cn) > 0:
            any_pos = True
    if not any_pos:
        return [_zero_like_param_dict(model, device=device) for _ in range(m)]

    noises = [{n: None for n in names} for _ in range(m)]
    for n, p in _all_named_params(model):
        cn = _canon_name(n)
        sig = _std_for(cn)
        if sig <= 0:
            for k in range(m):
                noises[k][cn] = torch.zeros_like(p, device=device or p.device)
            continue
        zs = [_randn_like_with_generator(p.to(device or p.device), sig, rng) for _ in range(m)]
        z_mean = torch.stack(zs, dim=0).mean(dim=0)
        scale = math.sqrt(m / (m - 1))
        for k in range(m):
            noises[k][cn] = (zs[k] - z_mean) * scale
    return noises

def _resolve_inner_trainer(handler_name: str):
    name = handler_name.strip()
    module_map = {
        "GradAscent": "trainer.unlearn.grad_ascent",
        "GradDiff": "trainer.unlearn.grad_diff",
        "NPO": "trainer.unlearn.npo",
        "DPO": "trainer.unlearn.dpo",
        "SimNPO": "trainer.unlearn.simnpo",
        "RMU": "trainer.unlearn.rmu",
        "UNDIAL": "trainer.unlearn.undial",
        "CEU": "trainer.unlearn.ceu",
        "SatImp": "trainer.unlearn.satimp",
        "WGA": "trainer.unlearn.wga",
        "PDU": "trainer.unlearn.pdu",
    }
    if name not in module_map:
        raise NotImplementedError(f"Unsupported inner handler '{name}' for PUM")
    mod = importlib.import_module(module_map[name])
    cls = getattr(mod, name, None)
    if cls is None:
        raise RuntimeError(f"Failed to resolve class '{name}' in module '{module_map[name]}'")
    return cls


# ----------------------------
# PUM Trainer
# ----------------------------

class PUM(FinetuneTrainer):
    """
    Perturb–Unlearn–Merge meta-trainer:
      • Public-only quantile C_ℓ → Δ̄_{2,ℓ}=2C_ℓ → RDP-calibrated σ (per-layer or single)
      • Zero-sum correlated noise across copies with α-scaling, tiny jitter
      • Optional function-preserving T: orthogonal head rotation + FFN permutation
      • EMA center from published means; server center-clipping
      • Multi-GPU safe: gather full unsharded state for diffs under ZeRO/Accelerate
    """

    def __init__(
        self,
        # PUM-specific
        inner_handler: str = "NPO",
        inner_method_args: Optional[dict] = None,
        copies_m: int = 4,
        rounds_R: int = 1,
        sigma: float = 0.0,
        sigma_per_layer: Optional[List[float]] = None,
        per_layer_noise: bool = False,
        # DP
        dp_epsilon: Optional[float] = None,
        dp_delta: Optional[float] = None,
        dp_sensitivity_total_l2: Optional[float] = None,
        dp_sensitivity_per_layer_l2: Optional[List[float]] = None,
        dp_rdp_orders: Optional[List[float]] = None,
        dp_use_worstcase_alpha: bool = True,
        dp_per_layer_allocation: str = "auto",
        alpha_min: float = 1.0,
        alpha_max: float = 1.1,
        eta_srv: float = 1.0,
        # Center clipping + EMA ref
        theta_ref_beta: float = 0.8,
        server_center_clipping: Optional[bool] = None,
        center_clip_C_global: Optional[float] = None,
        center_clip_C_per_layer: Optional[List[float]] = None,
        center_clip_quantile_q: float = 0.95,
        center_clip_quantile_kappa: float = 1.25,
        center_clip_round_gamma: float = 1.3,
        center_clip_ref_model_paths: Optional[List[str]] = None,
        # Jitter
        jitter_tau: float = 0.0,
        jitter_rel_to_sigma: float = 1e-4,
        # Local budget
        local_epochs: int = 1,
        local_max_steps: Optional[int] = None,
        auto_balance_local_max_steps: bool = True,
        # Safeguard clipping
        clip_update_norm: Optional[float] = None,
        clip_update_norm_per_layer: Optional[List[float]] = None,
        # Reparam
        use_orthogonal_reparam: bool = False,
        # Synthetic public fallback for C_ℓ
        center_clip_ref_synth_J: int = 8,
        synth_disable_attn: bool = False,
        synth_attn_eps: float = 0.03,
        synth_ffn_perm_frac: float = 0.25,

        # NEW: bounded synthetic calibration & clamps
        synth_mode: str = "gaussian",           # "gaussian" | "function_preserving"
        synth_gauss_rho: float = 0.02,          # C_ℓ ≈ κ·ρ·RMS(θ_ref,ℓ)
        c_rel_clip: Optional[float] = None,     # cap C_ℓ ≤ c_rel_clip·RMS(θ_ref,ℓ)
        c_rel_floor: Optional[float] = None,    # floor C_ℓ ≥ c_rel_floor·RMS(θ_ref,ℓ)
        sigma_warmup_factor: Optional[float] = None,
        sigma_min_rel: float = 0.0,             # σ ≥ sigma_min_rel·RMS(θ_ref)

        # base trainer args
        *args,
        **kwargs,
    ):
        method_args_override = kwargs.pop("method_args", None)
        super().__init__(*args, **kwargs)

        def _as_bool(x, default=False):
            if isinstance(x, bool) or x is None:
                return bool(x) if x is not None else default
            if isinstance(x, str):
                s = x.strip().lower()
                if s in ("true","1","yes","y"): return True
                if s in ("false","0","no","n","null","none",""): return False
            return bool(x)

        def _as_list_numbers(x):
            if x is None: return None
            if isinstance(x, (list, tuple)): return [float(v) for v in x]
            if isinstance(x, str):
                s = x.strip()
                if s.lower() in ("null","none",""): return None
                import ast
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, (list, tuple)): return [float(u) for u in v]
                except Exception: pass
                try: return [float(u) for u in s.split(",") if u.strip()]
                except Exception: return None
            try: return [float(x)]
            except Exception: return None

        def _as_list_str(x):
            if x is None: return None
            if isinstance(x, (list, tuple)): return [str(v) for v in x]
            if isinstance(x, str):
                s = x.strip()
                if s.lower() in ("null","none",""): return None
                import ast
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, (list, tuple)): return [str(u) for u in v]
                except Exception: pass
                if "," in s: return [u.strip() for u in s.split(",") if u.strip()]
                return [s]
            return [str(x)]

        # Assign
        self.inner_handler = inner_handler
        self.inner_method_args = dict(inner_method_args or {})
        self.copies_m = int(copies_m)
        self.rounds_R = int(rounds_R)
        self.sigma = float(sigma)
        self.sigma_per_layer = list(sigma_per_layer) if sigma_per_layer is not None else None
        self.per_layer_noise = _as_bool(per_layer_noise, default=False)

        # DP
        self.dp_epsilon = float(dp_epsilon) if dp_epsilon is not None else None
        self.dp_delta = float(dp_delta) if dp_delta is not None else None
        self.dp_sens_tot_l2 = float(dp_sensitivity_total_l2) if dp_sensitivity_total_l2 is not None else None
        self.dp_sens_per_layer_l2 = list(dp_sensitivity_per_layer_l2) if dp_sensitivity_per_layer_l2 is not None else None
        self.dp_rdp_orders = _as_list_numbers(dp_rdp_orders)
        self.dp_use_worstcase_alpha = _as_bool(dp_use_worstcase_alpha, default=True)
        dpi = str(dp_per_layer_allocation).lower().strip()
        self.dp_per_layer_allocation = dpi if dpi in ("auto","equalized","varmin") else "auto"
        self.alpha_min = max(float(alpha_min), 1.0)
        self.alpha_max = max(float(alpha_max), self.alpha_min)
        self.eta_srv = float(eta_srv)
        self.theta_ref_beta = float(theta_ref_beta)

        # Center clipping
        self.center_clip_C_global = float(center_clip_C_global) if center_clip_C_global is not None else None
        self.center_clip_C_per_layer = list(center_clip_C_per_layer) if center_clip_C_per_layer is not None else None
        self.center_clip_quantile_q = float(center_clip_quantile_q)
        self.center_clip_quantile_kappa = float(center_clip_quantile_kappa)
        self.center_clip_round_gamma = float(center_clip_round_gamma)
        self.center_clip_ref_model_paths = _as_list_str(center_clip_ref_model_paths)

        # Jitter / warmup
        self.jitter_tau = float(jitter_tau)
        self.jitter_rel_to_sigma = float(jitter_rel_to_sigma)
        if sigma_warmup_factor is None:
            v = os.environ.get("PUM_SIGMA_WARMUP_FACTOR", "").strip()
            try: sigma_warmup_factor = float(v) if v else 1.0
            except Exception: sigma_warmup_factor = 1.0
        self.sigma_warmup_factor = max(float(sigma_warmup_factor), 0.0)

        # Local budget + guardrails
        self.local_epochs = int(local_epochs)
        self.local_max_steps = int(local_max_steps) if local_max_steps is not None else None
        self.auto_balance_local_max_steps = _as_bool(auto_balance_local_max_steps, default=True)
        self.clip_update_norm = clip_update_norm
        self.clip_update_norm_per_layer = list(clip_update_norm_per_layer) if clip_update_norm_per_layer is not None else None

        # Reparam + synth refs
        self.use_orthogonal_reparam = _as_bool(use_orthogonal_reparam, default=False)
        self.center_clip_ref_synth_J = max(0, int(center_clip_ref_synth_J))
        self.synth_disable_attn = _as_bool(synth_disable_attn, default=False)
        self.synth_attn_eps = float(synth_attn_eps)
        self.synth_ffn_perm_frac = max(0.0, min(1.0, float(synth_ffn_perm_frac)))

        # NEW: bounded synthetic calibration config (env overrides)
        synth_mode_env = os.environ.get("PUM_SYNTH_MODE", "").strip().lower()
        if synth_mode_env in ("gaussian","gauss"):
            self.synth_mode = "gaussian"
        elif synth_mode_env in ("function_preserving","fp"):
            self.synth_mode = "function_preserving"
        else:
            self.synth_mode = str(synth_mode).strip().lower() if synth_mode else "gaussian"
        try:
            v = os.environ.get("PUM_SYNTH_GAUSS_RHO", "").strip()
            self.synth_gauss_rho = float(v) if v else float(synth_gauss_rho)
        except Exception:
            self.synth_gauss_rho = float(synth_gauss_rho)
        try:
            v = os.environ.get("PUM_C_REL_CLIP", "").strip()
            self.c_rel_clip = float(v) if v else (float(c_rel_clip) if c_rel_clip is not None else None)
        except Exception:
            self.c_rel_clip = c_rel_clip
        try:
            v = os.environ.get("PUM_C_REL_FLOOR", "").strip()
            self.c_rel_floor = float(v) if v else (float(c_rel_floor) if c_rel_floor is not None else None)
        except Exception:
            self.c_rel_floor = c_rel_floor
        try:
            v = os.environ.get("PUM_SIGMA_MIN_REL", "").strip()
            self.sigma_min_rel = float(v) if v else float(sigma_min_rel)
        except Exception:
            self.sigma_min_rel = float(sigma_min_rel)

        if self.copies_m < 1:
            raise ValueError("copies_m must be >= 1")

        # Nested overrides via Hydra `method_args`
        if isinstance(method_args_override, dict):
            _known = {
                "inner_handler","inner_method_args","copies_m","rounds_R",
                "sigma","sigma_per_layer","per_layer_noise",
                "dp_epsilon","dp_delta","dp_sensitivity_total_l2","dp_sensitivity_per_layer_l2",
                "dp_rdp_orders","dp_use_worstcase_alpha","dp_per_layer_allocation",
                "alpha_min","alpha_max","eta_srv",
                "theta_ref_beta","server_center_clipping",
                "center_clip_C_global","center_clip_C_per_layer","center_clip_quantile_q",
                "center_clip_quantile_kappa","center_clip_round_gamma","center_clip_ref_model_paths",
                "jitter_tau","jitter_rel_to_sigma",
                "local_epochs","local_max_steps","auto_balance_local_max_steps",
                "clip_update_norm","clip_update_norm_per_layer",
                "use_orthogonal_reparam",
                "center_clip_ref_synth_J",
                # new keys
                "synth_mode","synth_gauss_rho","c_rel_clip","c_rel_floor",
                "sigma_warmup_factor","sigma_min_rel",
            }
            for k, v in method_args_override.items():
                if k in _known:
                    if k in {"sigma_per_layer","dp_rdp_orders","center_clip_C_per_layer",
                             "dp_sensitivity_per_layer_l2","center_clip_ref_model_paths"}:
                        setattr(self, k, v if isinstance(v, list) else (v if v is None else [v]))
                    else:
                        setattr(self, k, v)
                elif k == "inner":
                    self.inner_handler = str(v)
                elif k == "inner_method_args" and isinstance(v, dict):
                    self.inner_method_args.update(v)

        # Test caps
        if PUM_FORCE_MAX_ROUNDS and PUM_FORCE_MAX_ROUNDS > 0:
            self.rounds_R = min(self.rounds_R, PUM_FORCE_MAX_ROUNDS)
        if PUM_FORCE_MAX_COPIES and PUM_FORCE_MAX_COPIES > 0:
            self.copies_m = min(self.copies_m, PUM_FORCE_MAX_COPIES)
        if PUM_FORCE_LOCAL_MAX_STEPS is not None and PUM_FORCE_LOCAL_MAX_STEPS > 0:
            self.local_max_steps = PUM_FORCE_LOCAL_MAX_STEPS
            self.auto_balance_local_max_steps = False

        # Seed & shape
        self._base_seed = int(self.args.seed or 0)
        try:
            self._num_layers = int(len(self.model.model.layers))  # type: ignore[attr-defined]
        except Exception:
            self._num_layers = None

        if self.sigma_per_layer is not None and self._num_layers is not None and len(self.sigma_per_layer) != self._num_layers:
            logger.warning("PUM: sigma_per_layer length=%d mismatches model layers=%s; using overlap.",
                           len(self.sigma_per_layer), str(self._num_layers))
        if self.clip_update_norm_per_layer is not None and self._num_layers is not None and len(self.clip_update_norm_per_layer) != self._num_layers:
            logger.warning("PUM: clip_update_norm_per_layer length=%d mismatches model layers=%s; using overlap.",
                           len(self.clip_update_norm_per_layer), str(self._num_layers))

        # Enable server clipping by default if any thresholds/sensitivities/synth refs exist
        if server_center_clipping is None:
            self.server_center_clipping = (
                (self.dp_sens_per_layer_l2 is not None and len(self.dp_sens_per_layer_l2) > 0)
                or (self.dp_sens_tot_l2 is not None and self.dp_sens_tot_l2 > 0)
                or (self.center_clip_C_per_layer is not None and len(self.center_clip_C_per_layer) > 0)
                or (self.center_clip_C_global is not None and self.center_clip_C_global > 0)
                or (self.center_clip_ref_model_paths is not None and len(self.center_clip_ref_model_paths) > 0)
                or (self.center_clip_ref_synth_J > 0)
            )
        else:
            self.server_center_clipping = _as_bool(server_center_clipping, default=False)

        # Base & reference state dicts
        self._theta_ref_sd: Dict[str, torch.Tensor] = self._state_dict_tensors(self.model)   # DP-safe ref (EMA of means)
        self._theta_base_sd: Dict[str, torch.Tensor] = {k: v.clone().detach().cpu() for k, v in self._theta_ref_sd.items()}
        # Pre-compute base layer norms (for logging / fallback only)
        self._layer_l2: Optional[List[float]] = self._compute_layer_l2_from_sd(self._theta_base_sd)
        self._layer_rms: Optional[List[float]] = self._compute_layer_rms_from_sd(self._theta_base_sd)

        # Global RMS (base)
        self._base_global_rms = self._global_rms_from_sd(self._theta_base_sd)
        self._pub_mean_prev_sd: Optional[Dict[str, torch.Tensor]] = None
        self._center_clip_C_from_quantile: Optional[List[float]] = None

        # Guardrail notice
        if (not self.dp_epsilon or not self.dp_delta) and (self.sigma <= 0.0) and (self.sigma_per_layer is None) and \
           (self.center_clip_ref_model_paths in (None, []) and self.center_clip_ref_synth_J <= 0):
            logger.warning(
                "PUM: No DP targets and no reference models for public-only calibration; sigma==0. "
                "Provide public refs or enable synthetic refs (center_clip_ref_synth_J>0)."
            )

    # ---------- state dict helpers ----------
    def _state_dict_tensors(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {_canon_name(n): p.detach().clone().cpu() for (n, p) in _all_named_params(model)}

    def _compute_layer_l2_from_sd(self, sd: Dict[str, torch.Tensor]) -> Optional[List[float]]:
        if not self._num_layers or self._num_layers <= 0:
            return None
        L = int(self._num_layers)
        out = [0.0 for _ in range(L)]
        for name, v in sd.items():
            li = self._param_layer_index(name)
            if li is None or li >= L:
                continue
            out[li] += v.detach().float().pow(2).sum().item()
        return [math.sqrt(max(x, 0.0)) for x in out]

    def _compute_layer_rms_from_sd(self, sd: Dict[str, torch.Tensor]) -> Optional[List[float]]:
        """Compute per-layer RMS(θ,ℓ) for an arbitrary state dict (θ may be θ_base or θ_ref of any round)."""
        if not self._num_layers or self._num_layers <= 0:
            return None
        L = int(self._num_layers)
        sum_sq = [0.0 for _ in range(L)]
        counts = [0 for _ in range(L)]
        for name, v in sd.items():
            li = self._param_layer_index(name)
            if li is None or li >= L:
                continue
            t = v.detach().float()
            sum_sq[li] += t.pow(2).sum().item()
            counts[li] += t.numel()
        rms = []
        for li in range(L):
            c = max(counts[li], 1)
            rms.append(math.sqrt(max(sum_sq[li] / float(c), 0.0)))
        return rms

    def _global_rms_from_sd(self, sd: Dict[str, torch.Tensor]) -> float:
        denom = sum(v.numel() for v in sd.values())
        if denom <= 0:
            return 0.0
        num = sum(v.detach().float().pow(2).sum().item() for v in sd.values())
        return math.sqrt(max(num / max(denom, 1), 0.0))

    @staticmethod
    def _accelerate_get_full_state_dict(trainer_like) -> Optional[Dict[str, torch.Tensor]]:
        acc = getattr(trainer_like, "accelerator", None)
        if acc is None:
            return None
        try:
            sd = acc.get_state_dict(trainer_like.model)
            out = {}
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    out[_canon_name(k)] = v.detach().clone().cpu()
            return out
        except Exception as e:
            logger.warning("PUM: accelerator.get_state_dict failed; fallback to ZeRO gather if available: %s", str(e))
            return None

    @staticmethod
    def _deepspeed_gather_state_dict(model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        try:
            import deepspeed
            params = list(model.parameters())
            with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
                out = {}
                for n, p in model.named_parameters():
                    if p is not None:
                        out[_canon_name(n)] = p.detach().clone().cpu()
                return out
        except Exception as e:
            logger.warning("PUM: deepspeed gather failed: %s", str(e))
            return None

    def _full_state_dict_from_inner(self, inner_trainer) -> Dict[str, torch.Tensor]:
        sd = self._accelerate_get_full_state_dict(inner_trainer)
        if sd is not None and len(sd) > 0:
            return sd
        try:
            m_eff = inner_trainer.model
            acc = getattr(inner_trainer, "accelerator", None)
            if acc is not None:
                try:
                    m_eff = acc.unwrap_model(m_eff)
                except Exception:
                    pass
            sd2 = self._deepspeed_gather_state_dict(m_eff)
            if sd2 is not None and len(sd2) > 0:
                return sd2
        except Exception:
            pass
        logger.warning("PUM: Using direct state_dict() as fallback (may be partial under ZeRO).")
        try:
            raw = inner_trainer.model.state_dict()
            out = {}
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    out[_canon_name(k)] = v.detach().clone().cpu()
            return out
        except Exception:
            return {}

    # ----------------------------
    # DP calibration for σ
    # ----------------------------
    def _resolve_S_alpha(self, m: int) -> float:
        a_min = max(float(self.alpha_min), 1.0)
        a_max = max(float(self.alpha_max), a_min)
        if self.dp_use_worstcase_alpha or a_max <= a_min:
            return m / (a_min ** 2)
        try:
            return m * ((1.0 / a_min - 1.0 / a_max) / (a_max - a_min))
        except ZeroDivisionError:
            return m / (a_min ** 2)

    def _orders_grid(self) -> List[float]:
        if not self.dp_rdp_orders:
            # Broadened grid: small λ near 1 tighten δ term; larger λ helpful when Δ small.
            return [1.1, 1.25, 1.5, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256]
        orders = [float(x) for x in self.dp_rdp_orders if float(x) > 1.0]
        return orders or [1.5, 2.0, 4.0, 8.0, 16.0]

    @staticmethod
    def _param_layer_index(param_name: str) -> Optional[int]:
        prefix = "model.layers."
        if not param_name.startswith(prefix):
            return None
        rest = param_name[len(prefix):]
        try:
            dot = rest.find(".")
            if dot <= 0:
                return None
            return int(rest[:dot])
        except Exception:
            return None

    def _get_per_layer_sens(self, L: int) -> Optional[List[float]]:
        if L is None or L <= 0:
            return None
        if self._center_clip_C_from_quantile is not None and len(self._center_clip_C_from_quantile) > 0:
            vals = [2.0 * float(c) for c in self._center_clip_C_from_quantile[:L]]
            if len(vals) < L:
                vals += [float(vals[-1])] * (L - len(vals))
            return vals
        if self.dp_sens_per_layer_l2 is not None and len(self.dp_sens_per_layer_l2) > 0:
            vals = [float(x) for x in self.dp_sens_per_layer_l2[:L]]
            if len(vals) < L:
                vals += [float(vals[-1])] * (L - len(vals))
            return vals
        if self.clip_update_norm_per_layer is not None and len(self.clip_update_norm_per_layer) > 0:
            vals = [(float(x) if (x is not None and float(x) > 0) else 0.0) for x in self.clip_update_norm_per_layer[:L]]
            if len(vals) < L:
                vals += [0.0] * (L - len(vals))
            return vals
        if self.dp_sens_tot_l2 is not None and self.dp_sens_tot_l2 > 0 and L > 0:
            per = float(self.dp_sens_tot_l2) / math.sqrt(float(L))
            return [per for _ in range(L)]
        return None

    
    def _apply_sigma_safety_clip(self) -> None:
        """
        Cap/floor σ relative to RMS(θ_ref). When DP is ON and PUM_DP_STRICT=1 (default),
        we do NOT apply any upper cap that would reduce the DP-calibrated σ; we only log it.
        Floors are always applied (privacy-safe).
        Also logs detailed pre/post values and per-layer limits if PUM_LOG_SIGMA_DETAIL=1.
        """
        # ----- config -----
        try:
            sigma_rel_clip = float(os.environ.get("PUM_SIGMA_REL_CLIP", "0.25"))
        except (TypeError, ValueError):
            sigma_rel_clip = 0.25
        sigma_rel_clip = max(sigma_rel_clip, 0.0)

        min_rel = float(getattr(self, "sigma_min_rel", 0.0) or 0.0)
        dp_on = (self.dp_epsilon is not None and self.dp_delta is not None)
        dp_strict = bool(int(os.environ.get("PUM_DP_STRICT", "1"))) if dp_on else False
        log_detail = os.environ.get("PUM_LOG_SIGMA_DETAIL", "0").strip() not in ("", "0", "false", "False")

        # ----- reference stats (current round) -----
        cur_rms_global = self._global_rms_from_sd(self._theta_ref_sd) if hasattr(self, "_theta_ref_sd") else self._base_global_rms
        layer_rms_list = self._compute_layer_rms_from_sd(self._theta_ref_sd) if hasattr(self, "_theta_ref_sd") else self._layer_rms

        # Track pre/post for logging
        pre_scalar = float(self.sigma or 0.0)
        pre_list = [float(s or 0.0) for s in (self.sigma_per_layer or [])]

        # ----- build per-layer limits -----
        limits = None
        if self.sigma_per_layer is not None and layer_rms_list is not None:
            L = min(len(self.sigma_per_layer), len(layer_rms_list))
            limits = [sigma_rel_clip * max(float(layer_rms_list[i]), 1e-8) for i in range(L)]
        elif self.sigma_per_layer is not None and cur_rms_global > 0.0:
            limits = [sigma_rel_clip * max(cur_rms_global, 1e-8) for _ in self.sigma_per_layer]
        elif cur_rms_global > 0.0:
            limits = sigma_rel_clip * max(cur_rms_global, 1e-8)  # scalar limit

        # ----- apply (or skip) the upper cap -----
        clipped_count = 0
        clipped_examples = []  # (i, pre, limit, post, ratio)

        def _maybe_log_detail():
            if not log_detail:
                return
            if isinstance(limits, list) and self.sigma_per_layer is not None:
                ratios = []
                for i, (pre, lim, post) in enumerate(zip(pre_list[:len(limits)], limits, self.sigma_per_layer[:len(limits)])):
                    if lim <= 0: continue
                    r = pre / lim
                    if post < pre:  # actually clipped
                        ratios.append(r)
                        clipped_examples.append((i, pre, lim, post, r))
                ratios.sort()
                if ratios:
                    import statistics as _stats
                    logger.info("PUM σ-clip summary: clipped %d/%d layers; ratio(pre/limit) median=%.3g, max=%.3g",
                                len(ratios), len(limits), _stats.median(ratios), max(ratios))
                else:
                    logger.info("PUM σ-clip summary: no layers clipped by the upper cap.")
                # Print up to the worst 8
                clipped_examples.sort(key=lambda t: t[4], reverse=True)
                for (i, pre, lim, post, r) in clipped_examples[:8]:
                    logger.info("PUM σ-clip detail: layer %d: σ_DP=%.6g, limit=%.6g, used=%.6g, ratio=%.3f%s",
                                i, pre, lim, post, r, " [CLIPPED]" if post < pre else "")
            else:
                # scalar σ
                if isinstance(limits, float) and limits > 0:
                    post = float(self.sigma or 0.0)
                    r = (pre_scalar / limits) if limits > 0 else float("inf")
                    msg = " [CLIPPED]" if post < pre_scalar else ""
                    logger.info("PUM σ-clip summary: scalar σ_DP=%.6g, limit=%.6g, used=%.6g, ratio=%.3f%s",
                                pre_scalar, limits, post, r, msg)

        # Upper cap: only if NOT in DP-strict mode
        if sigma_rel_clip > 0.0 and limits is not None:
            if dp_on and dp_strict:
                # DP strict: do not reduce σ; only warn if it WOULD have clipped
                would_clip = False
                if isinstance(limits, list) and self.sigma_per_layer is not None:
                    for i in range(min(len(self.sigma_per_layer), len(limits))):
                        if self.sigma_per_layer[i] > limits[i]:
                            would_clip = True
                            break
                elif isinstance(limits, float):
                    if pre_scalar > limits:
                        would_clip = True
                if would_clip:
                    logger.warning(
                        "PUM(DP): upper σ cap (rel=%.3g) WOULD reduce the DP-calibrated σ. "
                        "This would weaken the DP guarantee. Cap skipped because PUM_DP_STRICT=1.",
                        sigma_rel_clip
                    )
                # leave σ as-is (no upper clipping), then proceed to floors
            else:
                # Apply upper cap
                if self.sigma_per_layer is not None and isinstance(limits, list):
                    new_vals = []
                    for i, s in enumerate(self.sigma_per_layer[:len(limits)]):
                        s0 = float(s or 0.0)
                        s1 = min(max(s0, 0.0), limits[i])
                        if s1 < s0: clipped_count += 1
                        new_vals.append(s1)
                    self.sigma_per_layer = new_vals
                    if clipped_count > 0:
                        logger.info("PUM: per-layer σ clipped by rel %.3g to layer RMS (current ref).", sigma_rel_clip)
                elif isinstance(limits, list) and self.sigma_per_layer is not None:
                    # fallback already covered above
                    pass
                elif isinstance(limits, float):
                    s0 = float(self.sigma or 0.0)
                    s1 = min(max(s0, 0.0), limits)
                    if s1 < s0:
                        clipped_count = 1
                        logger.info("PUM: σ clipped to <= %.6g (rel clip=%.3g, ref_rms=%.6g)", limits, sigma_rel_clip, cur_rms_global)
                    self.sigma = s1

        # ----- apply min floor (always safe) -----
        if min_rel > 0.0:
            if self.sigma_per_layer is not None and layer_rms_list is not None:
                new_vals = []
                floored = False
                for idx, s in enumerate(self.sigma_per_layer):
                    floor_i = min_rel * max(float(layer_rms_list[idx] if idx < len(layer_rms_list) else 0.0), 1e-8)
                    s_val = float(s or 0.0)
                    if s_val < floor_i:
                        s_val = floor_i
                        floored = True
                    new_vals.append(s_val)
                self.sigma_per_layer = new_vals
                if floored:
                    logger.info("PUM: per-layer σ floored by rel %.3g to layer RMS (current ref).", min_rel)
            elif self.sigma_per_layer is not None and cur_rms_global > 0.0:
                floor = min_rel * cur_rms_global
                self.sigma_per_layer = [max(float(s or 0.0), floor) for s in self.sigma_per_layer]
                logger.info("PUM: per-layer σ floored to >= %.6g based on global RMS (current ref).", floor)
            elif cur_rms_global > 0.0:
                floor = min_rel * cur_rms_global
                s_val = float(self.sigma or 0.0)
                if s_val < floor:
                    self.sigma = floor
                logger.info("PUM: σ floored to >= %.6g (min rel=%.3g, ref_rms=%.6g)", floor, min_rel, cur_rms_global)



    def _calibrate_sigma_from_dp(self) -> None:
        """Set self.sigma or self.sigma_per_layer using RDP with Δ̄ from C_ℓ."""
        if self.dp_epsilon is None or self.dp_delta is None:
            self._apply_sigma_safety_clip()
            return

        eps_tgt = float(self.dp_epsilon)
        delta = float(self.dp_delta)
        R = max(int(self.rounds_R), 1)
        m = max(int(self.copies_m), 1)
        S_alpha = self._resolve_S_alpha(m)
        orders = self._orders_grid()
        calibrated = False

        # per-layer calibration
        if self.per_layer_noise:
            L = int(self._num_layers) if self._num_layers is not None else None
            if not L or L <= 0:
                logger.warning("PUM DP per-layer requested but model layers unknown; fallback to single σ.")
            else:
                Delta_l = self._get_per_layer_sens(L)
                if Delta_l is None:
                    logger.warning("PUM DP per-layer needs per-layer/total sensitivity; fallback to single σ.")
                else:
                    best_sigmas: Optional[List[float]] = None
                    best_total_var: Optional[float] = None
                    for lam in orders:
                        A = eps_tgt - (math.log(1.0 / delta) / (lam - 1.0))
                        if A <= 0: continue
                        K = (2.0 * A) / (R * lam * max(S_alpha, 1e-12))
                        if K <= 0: continue
                        # equalized
                        kappa = math.sqrt(float(L) / K)
                        sig_eq = [kappa * d for d in Delta_l]
                        tot_eq = (kappa * kappa) * sum(d * d for d in Delta_l)
                        # var-min
                        sum_D = sum(Delta_l)
                        c = math.sqrt(max(sum_D, 0.0) / K)
                        sig_vm = [c * math.sqrt(max(d, 0.0)) for d in Delta_l]
                        tot_vm = (c * c) * sum_D
                        for total_var, sigmas in ((tot_eq, sig_eq), (tot_vm, sig_vm)):
                            if not all(math.isfinite(s) and s >= 0 for s in sigmas):
                                continue
                            if (best_total_var is None) or (total_var < best_total_var):
                                best_total_var = total_var
                                best_sigmas = sigmas
                    if best_sigmas is not None:
                        self.sigma_per_layer = [float(s) for s in best_sigmas]
                        self.sigma = 0.0
                        calibrated = True
                        logger.info("PUM DP per-layer calibration: σ[0:3]=%s ...", str(self.sigma_per_layer[:3]))

        # single σ fallback
        if not calibrated:
            if self.dp_sens_tot_l2 is None:
                self._apply_sigma_safety_clip()
                return
            Delta2_tot = float(self.dp_sens_tot_l2)
            best_sigma = None
            for lam in orders:
                A = eps_tgt - (math.log(1.0 / delta) / (lam - 1.0))
                if A <= 0: continue
                sig = Delta2_tot * math.sqrt((R * lam * max(S_alpha, 1e-12)) / (2.0 * A))
                if sig <= 0 or math.isnan(sig) or math.isinf(sig): continue
                if (best_sigma is None) or (sig < best_sigma):
                    best_sigma = sig
            if best_sigma is None:
                logger.warning("PUM DP-calibration failed; keeping existing σ.")
                self._apply_sigma_safety_clip()
                return
            self.sigma = float(best_sigma)
            if self.per_layer_noise:
                self.sigma_per_layer = None
            calibrated = True
            logger.info("PUM DP single-sigma: σ=%.6g", self.sigma)

        if calibrated:
            self._apply_sigma_safety_clip()

    # ----------------------------
    # Server-side clipping: quantile C_ℓ (public-only) + EMA reference
    # ----------------------------
    def _resolve_center_clip_thresholds(self) -> Tuple[Optional[List[float]], Optional[float]]:
        L = int(self._num_layers) if self._num_layers is not None else None
        C_per_layer: Optional[List[float]] = None
        C_global: Optional[float] = None
        if self._center_clip_C_from_quantile is not None:
            # Avoid multiplicative drift of C_ℓ across rounds: hold the cached thresholds.
            C_per_layer = [float(c) for c in self._center_clip_C_from_quantile]
            return C_per_layer, None
        if self.center_clip_C_per_layer is not None and len(self.center_clip_C_per_layer) > 0:
            C_per_layer = [float(x) for x in self.center_clip_C_per_layer]
        elif self.dp_sens_per_layer_l2 is not None and len(self.dp_sens_per_layer_l2) > 0:
            C_per_layer = [0.5 * float(x) for x in self.dp_sens_per_layer_l2]
        elif self.dp_sens_tot_l2 is not None and L and L > 0:
            C_tot = 0.5 * float(self.dp_sens_tot_l2)
            per = C_tot / math.sqrt(float(L))
            C_per_layer = [per for _ in range(L)]
        if self.center_clip_C_global is not None:
            C_global = float(self.center_clip_C_global)
        elif self.dp_sens_tot_l2 is not None and (not C_per_layer):
            C_global = 0.5 * float(self.dp_sens_tot_l2)
        return C_per_layer, C_global

    def _materialize_params(self, model, state):
        with torch.no_grad():
            for n, p in _all_named_params(model):
                v = state.get(_canon_name(n))
                if v is not None:
                    p.copy_(v.to(device=p.device, dtype=p.dtype))

    def _sample_T(self, model: nn.Module, seed: int, force: bool = False):
        # force=True: only for function-preserving synthetic refs
        use_attn = (not force and self.use_orthogonal_reparam) or (force and not self.synth_disable_attn)
        use_ffn  = True
        torch.manual_seed(seed)
        T: Dict[int, Dict[str, torch.Tensor]] = {}
        try:
            layers = model.model.layers  # type: ignore[attr-defined]
        except Exception:
            return None
        for li, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            mlp = getattr(layer, "mlp", None)
            entry: Dict[str, torch.Tensor] = {}

            if attn is not None and use_attn:
                head_dim = int(getattr(attn, "head_dim", 0) or (attn.q_proj.weight.shape[0] //
                                                                getattr(attn, "num_heads", 1)))
                if head_dim > 0:
                    if force:
                        R = _near_identity_orthogonal(
                            head_dim,
                            eps=max(self.synth_attn_eps, 0.0),
                            device=attn.q_proj.weight.device,
                            dtype=attn.q_proj.weight.dtype,
                        )
                    else:
                        A = torch.randn(head_dim, head_dim, device="cpu", dtype=torch.float32)
                        Q, _ = torch.linalg.qr(A)
                        R = Q.detach().to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
                    entry["R_sub"] = R
                    entry["n_q_heads"] = torch.tensor(int(getattr(attn, "num_heads", 0)), device="cpu")
                    entry["n_kv_heads"] = torch.tensor(int(getattr(attn, "num_key_value_heads",
                                                                getattr(attn, "num_heads", 0))), device="cpu")
                    entry["head_dim"] = torch.tensor(head_dim, device="cpu")

            if mlp is not None and use_ffn:
                inter = int(mlp.up_proj.weight.shape[0])
                perm = torch.arange(inter, device=mlp.up_proj.weight.device)
                if force and self.synth_ffn_perm_frac < 1.0:
                    k = int(round(self.synth_ffn_perm_frac * inter))
                    if k > 1:
                        idx = torch.randperm(inter, device=perm.device)[:k]
                        perm_part = idx[torch.randperm(k, device=perm.device)]
                        perm[idx] = perm_part
                else:
                    perm = torch.randperm(inter, device=mlp.up_proj.weight.device)
                inv_perm = torch.empty_like(perm)
                inv_perm[perm] = torch.arange(inter, device=perm.device)
                entry["perm"] = perm
                entry["inv_perm"] = inv_perm

            if entry:
                T[li] = entry
        return T if len(T) > 0 else None

    def _apply_T(self, model: nn.Module, T) -> None:
        if not T: return
        try:
            layers = model.model.layers  # type: ignore[attr-defined]
        except Exception:
            return
        for li, layer in enumerate(layers):
            if li not in T: continue
            ent = T[li]
            attn = getattr(layer, "self_attn", None)
            mlp = getattr(layer, "mlp", None)
            if attn is not None and "R_sub" in ent:
                R = ent["R_sub"]
                D = int(ent["head_dim"].item())
                Hq = int(ent["n_q_heads"].item())
                Hkv = int(ent["n_kv_heads"].item())

                def _left_mul_block(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    w = weight.view(H, D, -1)
                    if w.device.type == "cpu":
                        w32 = w.float(); R32 = R.float()
                        w32 = torch.einsum("ij,hjm->him", R32, w32)
                        return w32.to(weight.dtype).reshape(H * D, -1)
                    else:
                        w = torch.einsum("ij,hjm->him", R, w)
                        return w.reshape(H * D, -1)

                def _right_mul_block(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    w = weight.view(-1, H, D)
                    if w.device.type == "cpu":
                        w32 = w.float(); R32 = R.float()
                        w32 = torch.einsum("mhd,dj->mhj", w32, R32.T)
                        return w32.to(weight.dtype).reshape(-1, H * D)
                    else:
                        w = torch.einsum("mhd,dj->mhj", w, R.T)
                        return w.reshape(-1, H * D)

                with torch.no_grad():
                    attn.q_proj.weight.copy_(_left_mul_block(attn.q_proj.weight, Hq, D, R))
                    attn.k_proj.weight.copy_(_left_mul_block(attn.k_proj.weight, Hkv, D, R))
                    attn.v_proj.weight.copy_(_left_mul_block(attn.v_proj.weight, Hkv, D, R))
                    attn.o_proj.weight.copy_(_right_mul_block(attn.o_proj.weight, Hq, D, R))

            if mlp is not None and "perm" in ent and "inv_perm" in ent:
                P = ent["perm"].long(); invP = ent["inv_perm"].long()
                with torch.no_grad():
                    mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[P, :])
                    mlp.up_proj.weight.copy_(mlp.up_proj.weight[P, :])
                    if getattr(mlp.gate_proj, "bias", None) is not None:
                        mlp.gate_proj.bias.copy_(mlp.gate_proj.bias[P])
                    if getattr(mlp.up_proj, "bias", None) is not None:
                        mlp.up_proj.bias.copy_(mlp.up_proj.bias[P])
                    mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, invP])

    def _apply_T_inv_to_update(self, update: Dict[str, torch.Tensor], T) -> Dict[str, torch.Tensor]:
        if not T: return update
        out: Dict[str, torch.Tensor] = dict(update)

        def _maybe(name: str):
            return name in out and isinstance(out[name], torch.Tensor)

        for li, ent in T.items():
            base = f"model.layers.{li}."
            if "R_sub" in ent:
                R = ent["R_sub"]
                D = int(ent["head_dim"].item())
                Hq = int(ent["n_q_heads"].item())
                Hkv = int(ent["n_kv_heads"].item())

                def _left_inv(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    w = weight.view(H, D, -1)
                    if w.device.type == "cpu":
                        w32 = w.float(); RT32 = R.float().T
                        w32 = torch.einsum("ij,hjm->him", RT32, w32)
                        return w32.to(weight.dtype).reshape(H * D, -1)
                    else:
                        w = torch.einsum("ij,hjm->him", R.T, w)
                        return w.reshape(H * D, -1)

                def _right_inv(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    w = weight.view(-1, H, D)
                    if w.device.type == "cpu":
                        w32 = w.float(); R32 = R.float()
                        w32 = torch.einsum("mhd,dj->mhj", w32, R32)
                        return w32.to(weight.dtype).reshape(-1, H * D)
                    else:
                        w = torch.einsum("mhd,dj->mhj", w, R)
                        return w.reshape(-1, H * D)

                q_w = base + "self_attn.q_proj.weight"
                k_w = base + "self_attn.k_proj.weight"
                v_w = base + "self_attn.v_proj.weight"
                o_w = base + "self_attn.o_proj.weight"
                if _maybe(q_w): out[q_w] = _left_inv(out[q_w], Hq, D, R)
                if _maybe(k_w): out[k_w] = _left_inv(out[k_w], Hkv, D, R)
                if _maybe(v_w): out[v_w] = _left_inv(out[v_w], Hkv, D, R)
                if _maybe(o_w): out[o_w] = _right_inv(out[o_w], Hq, D, R)

            if "perm" in ent and "inv_perm" in ent:
                P = ent["perm"].long()
                invP = ent["inv_perm"].long()
                g_w = base + "mlp.gate_proj.weight"
                u_w = base + "mlp.up_proj.weight"
                d_w = base + "mlp.down_proj.weight"
                g_b = base + "mlp.gate_proj.bias"
                u_b = base + "mlp.up_proj.bias"
                if _maybe(g_w): out[g_w] = out[g_w][invP, :]
                if _maybe(u_w): out[u_w] = out[u_w][invP, :]
                if _maybe(g_b): out[g_b] = out[g_b][invP]
                if _maybe(u_b): out[u_b] = out[u_b][invP]
                if _maybe(d_w): out[d_w] = out[d_w][:, P]

        return out

    # ----------------------------
    # Update clipping
    # ----------------------------
    def _clip_update(self, update: Dict[str, torch.Tensor]):
        if self.clip_update_norm_per_layer is not None and self._num_layers:
            L = min(int(self._num_layers), len(self.clip_update_norm_per_layer))
            layer_sq = [0.0 for _ in range(L)]
            for name, v in update.items():
                li = self._param_layer_index(name)
                if li is None or li >= L: continue
                layer_sq[li] += v.detach().float().pow(2).sum().item()
            scales = [1.0 for _ in range(L)]
            for li in range(L):
                thr = float(self.clip_update_norm_per_layer[li])
                if thr is None or thr <= 0: continue
                norm = math.sqrt(max(layer_sq[li], 1e-12))
                if norm > thr:
                    scales[li] = thr / (norm + 1e-12)
            out: Dict[str, torch.Tensor] = {}
            for name, v in update.items():
                li = self._param_layer_index(name)
                if li is not None and li < L:
                    s = scales[li]
                    out[name] = v * s if s != 1.0 else v
                else:
                    out[name] = v
            if self.clip_update_norm is not None:
                other_sq = 0.0
                for name, v in update.items():
                    if self._param_layer_index(name) is None or (self._param_layer_index(name) >= L):
                        other_sq += v.detach().float().pow(2).sum().item()
                other_norm = math.sqrt(max(other_sq, 1e-12))
                if other_norm > self.clip_update_norm:
                    s = self.clip_update_norm / (other_norm + 1e-12)
                    for name, v in update.items():
                        if self._param_layer_index(name) is None or (self._param_layer_index(name) >= L):
                            out[name] = v * s
            return out

        if self.clip_update_norm is not None and self.clip_update_norm > 0:
            total = 0.0
            for v in update.values():
                total += v.detach().float().pow(2).sum().item()
            norm = math.sqrt(max(total, 1e-12))
            if norm <= self.clip_update_norm:
                return update
            scale = self.clip_update_norm / (norm + 1e-12)
            return {k: v * scale for k, v in update.items()}
        return update

    # ----------------------------
    # Local steps auto-balance
    # ----------------------------
    def _maybe_set_auto_local_max_steps(self) -> None:
        if self.local_max_steps is not None or not self.auto_balance_local_max_steps:
            return
        try:
            dataset_len = len(self.train_dataset)
        except Exception:
            dataset_len = None
        if not dataset_len or dataset_len <= 0:
            logger.warning("PUM: cannot infer dataset length; skip auto-balance of local_max_steps")
            return
        bs = int(self.args.per_device_train_batch_size)
        accum = int(self.args.gradient_accumulation_steps or 1)
        try:
            world = int(getattr(self.args, "world_size", None) or int(os.environ.get("WORLD_SIZE", 1)))
        except Exception:
            world = 1
        explicit_max_steps = int(getattr(self.args, "max_steps", -1) or -1)
        if explicit_max_steps > 0:
            N = explicit_max_steps
        else:
            steps_per_epoch = math.ceil(dataset_len / max(bs * accum * world, 1))
            epochs = int(getattr(self.args, "num_train_epochs", 1))
            N = steps_per_epoch * max(epochs, 1)
        self.local_max_steps = max(1, math.ceil(N / max(self.rounds_R, 1)))
        logger.info("PUM auto-balance: N=%d, R=%d → local_max_steps=%d", N, self.rounds_R, self.local_max_steps)

    # ----------------------------
    # Main train loop
    # ----------------------------
    def _make_inner_args(self, output_dir: str) -> TrainingArguments:
        inner_args = copy.deepcopy(self.args)
        inner_args.output_dir = output_dir
        inner_args.save_strategy = "no"
        inner_args.eval_strategy = "no"
        inner_args.do_eval = False
        inner_args.logging_dir = os.path.join(output_dir, "logs")
        inner_args.report_to = [] if getattr(inner_args, "report_to", None) is None else inner_args.report_to
        inner_args.num_train_epochs = self.local_epochs
        if self.local_max_steps is not None:
            inner_args.max_steps = self.local_max_steps

        if (self.sigma > 0.0) or (self.sigma_per_layer is not None):
            for attr in ("bf16","fp16","torch_compile"):
                try: setattr(inner_args, attr, False)
                except Exception: pass

        inner_args.seed = (self._base_seed + 17) % (2**31 - 1)
        try: inner_args.data_seed = inner_args.seed
        except Exception: pass

        try:
            lr = float(getattr(inner_args, "learning_rate", 0.0))
        except Exception:
            lr = 0.0
        if not (lr > 0):
            default_lr = float(os.environ.get("PUM_INNER_LR", "1e-5"))
            inner_args.learning_rate = default_lr
            logger.warning("PUM(inner): learning_rate was 0/None; defaulting to %.2e", default_lr)

        if not getattr(inner_args, "lr_scheduler_type", None):
            inner_args.lr_scheduler_type = "constant"
            inner_args.warmup_steps = 0
            inner_args.warmup_ratio = 0.0

        return inner_args

    def _instantiate_inner_trainer(self, model: nn.Module, args: TrainingArguments):
        inner_cls = _resolve_inner_trainer(self.inner_handler)
        return inner_cls(
            model=model,
            train_dataset=self.train_dataset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=args,
            evaluators=None,
            template_args=self.template_args,
            **self.inner_method_args,
        )

    def _build_per_param_sigma(self) -> Optional[Dict[str, float]]:
        if self.sigma_per_layer is None:
            return None
        sigma_other = 0.0 if self.per_layer_noise else (self.sigma if (self.sigma and self.sigma > 0) else 0.0)
        per_param = {}
        for n, _ in _all_named_params(self.model):
            cn = _canon_name(n)
            li = self._param_layer_index(cn)
            per_param[cn] = float(self.sigma_per_layer[li]) if (li is not None and li < len(self.sigma_per_layer)) else float(sigma_other)
        return per_param

    def train(self, resume_from_checkpoint: Optional[str] = None, **kwargs):
        self.model.train()

        logger.info(
            "PUM config: inner=%s, m=%d, R=%d, per_layer_noise=%s, σ=%s, has_σ_per_layer=%s, server_clip=%s, synth_mode=%s, rho=%.4g",
            self.inner_handler, self.copies_m, self.rounds_R, str(self.per_layer_noise),
            f"{self.sigma:.6g}", str(self.sigma_per_layer is not None), str(self.server_center_clipping),
            self.synth_mode, float(self.synth_gauss_rho)
        )

        self._maybe_set_auto_local_max_steps()

        # Public-only C_ℓ (real or synthetic), then DP calibrate σ
        if self.server_center_clipping:
            self._maybe_init_center_C_from_quantile()
        self._calibrate_sigma_from_dp()

        for r in range(1, self.rounds_R + 1):
            # EMA ref from previous round
            if self._pub_mean_prev_sd is not None:
                beta = float(self.theta_ref_beta)
                with torch.no_grad():
                    for n, _ in _all_named_params(self.model):
                        cn = _canon_name(n)
                        prev_ref = self._theta_ref_sd.get(cn)
                        pub = self._pub_mean_prev_sd.get(cn, prev_ref)
                        if prev_ref is None:
                            if pub is not None:
                                self._theta_ref_sd[cn] = pub.detach().clone()
                        else:
                            if pub is not None:
                                self._theta_ref_sd[cn] = (1.0 - beta) * prev_ref + beta * pub

            # redo DP calib if C_ℓ updated
            self._calibrate_sigma_from_dp()

            # RNGs
            rng = torch.Generator(device="cpu"); rng.manual_seed(self._base_seed + 811 * r)
            g   = torch.Generator(device="cpu"); g.manual_seed(self._base_seed + 9973 * r)

            # per-parameter σ
            per_param_sigma = self._build_per_param_sigma()

            # warmup for round 1
            warm = float(self.sigma_warmup_factor if (r == 1 and self.sigma_warmup_factor is not None) else 1.0)
            if warm < 1.0:
                if per_param_sigma is not None:
                    per_param_sigma = {k: float(v) * warm for k, v in per_param_sigma.items()}
                else:
                    self.sigma = float(self.sigma) * warm
                logger.info("PUM: round-%d σ warmup factor applied: %.3g", r, warm)

            # zero-sum noises
            base_noises = _generate_zero_sum_noises(
                self.model, self.copies_m, self.sigma, per_param_sigma=per_param_sigma, rng=rng
            )

            # secret alphas
            alphas: List[float] = []
            for _ in range(self.copies_m):
                if self.alpha_max == self.alpha_min:
                    alphas.append(self.alpha_min)
                else:
                    u = torch.rand(1, generator=g).item()
                    alphas.append(self.alpha_min + u * (self.alpha_max - self.alpha_min))

            # accumulators
            S0 = 0.0
            S1 = _zero_like_param_dict(self.model)
            pub_sum_server = _zero_like_param_dict(self.model)
            cur_sd = self._state_dict_tensors(self.model)

            round_norms_per_layer: Optional[List[List[float]]] = (
                [[] for _ in range(int(self._num_layers))]
                if (self._num_layers is not None and int(self._num_layers) > 0)
                else None
            )

            center_delta = None
            if self.server_center_clipping:
                C_per_layer, C_global = self._resolve_center_clip_thresholds()
                center_delta = self._compute_center_clip_delta(cur_sd, self._theta_ref_sd, C_per_layer, C_global)

            # copies
            for k in range(self.copies_m):
                alpha_k = float(alphas[k])
                eps_k = {n: alpha_k * base_noises[k][n] for n in base_noises[k]}

                xi_k = None
                if self.jitter_rel_to_sigma > 0:
                    xi_k = {}
                    def _sig_for(cn: str) -> float:
                        if per_param_sigma is not None:
                            return float(per_param_sigma.get(cn, 0.0))
                        return float(self.sigma or 0.0)
                    for n, p in _all_named_params(self.model):
                        cn = _canon_name(n)
                        tau_n = self.jitter_rel_to_sigma * _sig_for(cn)
                        xi_k[cn] = _randn_like_with_generator(p, tau_n, rng) if tau_n > 0 else torch.zeros_like(p)

                model_k = copy.deepcopy(self.model)
                T_k = self._sample_T(model_k, seed=self._base_seed + 31 * r + 7 * k) if self.use_orthogonal_reparam else None

                if center_delta is not None:
                    _apply_state_dict_delta(model_k, center_delta, scale=1.0)
                _apply_state_dict_delta(model_k, eps_k, scale=1.0)
                if xi_k is not None:
                    _apply_state_dict_delta(model_k, xi_k, scale=1.0)

                self._apply_T(model_k, T_k)

                inner_out_dir = os.path.join(str(self.args.output_dir), f"pum_round{r}_copy{k+1}")
                inner_args = self._make_inner_args(inner_out_dir)
                inner_trainer = self._instantiate_inner_trainer(model_k, inner_args)

                # before
                before_sd_full = self._full_state_dict_from_inner(inner_trainer)
                before_sd_aligned = self._apply_T_inv_to_update(before_sd_full, T_k)

                with torch.no_grad():
                    for n in pub_sum_server.keys():
                        if n in before_sd_aligned:
                            clean = torch.nan_to_num(before_sd_aligned[n], nan=0.0, posinf=0.0, neginf=0.0)
                            pub_sum_server[n].add_(clean.to(pub_sum_server[n].device))

                if round_norms_per_layer is not None:
                    for name, base_v in self._theta_ref_sd.items():
                        li = self._param_layer_index(name)
                        if li is None or li >= len(round_norms_per_layer): continue
                        v = before_sd_aligned.get(name, None)
                        if v is None: continue
                        dv = (v.detach().cpu() - base_v).float().view(-1)
                        nrm = float(torch.linalg.norm(dv, ord=2).item())
                        round_norms_per_layer[li].append(nrm)

                # train
                inner_trainer.train()

                # merge PEFT if present
                try:
                    from peft import PeftModel
                    m_eff = inner_trainer.model
                    acc = getattr(inner_trainer, "accelerator", None)
                    if acc is not None:
                        try: m_eff = acc.unwrap_model(m_eff)
                        except Exception: pass
                    if isinstance(m_eff, PeftModel):
                        m_eff.merge_and_unload()
                        logger.info("[PUM] LoRA/PEFT adapters merged into base weights before diff.")
                        inner_trainer.model = m_eff
                except Exception as e:
                    logger.warning("[PUM] merge-and-unload adapters failed or PEFT not installed: %s", str(e))

                # after → delta
                after_sd_full = self._full_state_dict_from_inner(inner_trainer)
                delta_k = _diff_state_dict(after_sd_full, before_sd_full)
                delta_k = self._apply_T_inv_to_update(delta_k, T_k)

                overlap_norm_sq = 0.0
                nonoverlap_norm_sq = 0.0
                overlap_cnt = 0
                for n, dv in delta_k.items():
                    if n in S1:
                        overlap_cnt += 1
                        overlap_norm_sq += dv.detach().float().pow(2).sum().item()
                    else:
                        nonoverlap_norm_sq += dv.detach().float().pow(2).sum().item()
                logger.info(
                    "[PUM][round %d copy %d] overlap_params=%d, ||Δ_overlap||_2=%.3e, ||Δ_nonoverlap||_2=%.3e",
                    r, k+1, overlap_cnt, math.sqrt(max(overlap_norm_sq, 0.0)), math.sqrt(max(nonoverlap_norm_sq, 0.0))
                )

                delta_k = self._clip_update(delta_k)

                bad = False
                for name, tensor in delta_k.items():
                    cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                    if not torch.isfinite(cleaned).all():
                        bad = True; break
                    delta_k[name] = cleaned
                if bad:
                    logger.warning("PUM: skipping copy %d (round %d) due to non-finite delta.", k + 1, r)
                    del inner_trainer, model_k, after_sd_full, delta_k
                    torch.cuda.empty_cache()
                    continue

                w = 1.0 / max(alpha_k, 1e-12)
                S0 += w
                for n, dv in delta_k.items():
                    if n in S1:
                        S1[n].add_(w * dv.to(S1[n].device))

                del inner_trainer, model_k, after_sd_full, delta_k
                torch.cuda.empty_cache()

            # publish mean update
            inv_S0 = 1.0 / max(S0, 1e-12)
            with torch.no_grad():
                total_sq = 0.0
                for n, p in _all_named_params(self.model):
                    cn = _canon_name(n)
                    if cn not in S1: continue
                    update = torch.nan_to_num(S1[cn] * inv_S0, nan=0.0, posinf=0.0, neginf=0.0).to(p.device, p.dtype)
                    total_sq += update.detach().float().pow(2).sum().item()
                    p.add_(self.eta_srv * update)
                logger.info("[PUM][round %d] ||mean update||_2 = %.6f", r, math.sqrt(max(total_sq, 0.0)))

            # next-round EMA center from mean(before)
            m_float = float(self.copies_m)
            self._pub_mean_prev_sd = {n: (pub_sum_server[n] / m_float).detach().clone() for n in pub_sum_server}

            # update C_ℓ from observed norms (drift‑safe, monotone non‑increasing)
            if round_norms_per_layer is not None:
                q = min(max(self.center_clip_quantile_q, 0.0), 1.0)
                kappa = max(self.center_clip_quantile_kappa, 0.0)
                Cs_obs: List[float] = []
                for li in range(len(round_norms_per_layer)):
                    vals = round_norms_per_layer[li]
                    if not vals:
                        Cs_obs.append(0.0)
                    else:
                        t = torch.tensor(vals, dtype=torch.float32)
                        c_l = float(torch.quantile(t, q).item()) * kappa
                        Cs_obs.append(c_l)
                # Apply relative clip/floor w.r.t current θ_ref RMS
                rms_ref = self._compute_layer_rms_from_sd(self._theta_ref_sd)
                if rms_ref is not None:
                    for li in range(len(Cs_obs)):
                        R_i = float(rms_ref[li]) if li < len(rms_ref) else 0.0
                        if self.c_rel_clip is not None and self.c_rel_clip > 0 and R_i > 0:
                            Cs_obs[li] = min(Cs_obs[li], self.c_rel_clip * R_i)
                        if self.c_rel_floor is not None and self.c_rel_floor > 0 and R_i > 0:
                            Cs_obs[li] = max(Cs_obs[li], self.c_rel_floor * R_i)
                # Monotone non‑increasing across rounds
                if self._center_clip_C_from_quantile is None:
                    self._center_clip_C_from_quantile = Cs_obs
                else:
                    prev = self._center_clip_C_from_quantile
                    L = min(len(prev), len(Cs_obs))
                    self._center_clip_C_from_quantile = [min(prev[i], Cs_obs[i]) for i in range(L)]
                    # keep tail if any
                    if len(prev) > L:
                        self._center_clip_C_from_quantile += prev[L:]
                    elif len(Cs_obs) > L:
                        self._center_clip_C_from_quantile += Cs_obs[L:]
                if len(self._center_clip_C_from_quantile) >= 3:
                    logger.info("PUM: updated (monotone) C_l [0:3]=%s ...", str(self._center_clip_C_from_quantile[:3]))

        self.model.eval()
        acc = getattr(self, "accelerator", None)
        is_main = True if (acc is None) else bool(acc.is_main_process)
        if is_main:
            try:
                self.model.save_pretrained(str(self.args.output_dir))
                if getattr(self, "tokenizer", None) is not None:
                    self.tokenizer.save_pretrained(str(self.args.output_dir))
            except Exception as e:
                logger.warning("PUM: failed to save_pretrained: %s", e)
        return None

    # ---------- quantile C_ℓ computation (public-only & synthetic fallback) ----------
    def _maybe_init_center_C_from_quantile(self) -> None:
        if self._center_clip_C_from_quantile is not None:
            return
        if not self._num_layers or self._num_layers <= 0:
            logger.warning("PUM: cannot compute quantile C_l; model layers unknown")
            return

        L = int(self._num_layers)
        norms_per_layer: List[List[float]] = [[] for _ in range(L)]

        def accumulate_from_state_dict(sd: Dict[str, torch.Tensor]):
            for name, base_v in self._theta_ref_sd.items():  # CRITICAL FIX: anchor at θ_ref (θ^{(j−1)}), not θ_base
                li = self._param_layer_index(name)
                if li is None or li >= L:
                    continue
                ref_v = sd.get(name, None)
                if ref_v is None:
                    continue
                dv = (ref_v.detach().cpu() - base_v).float().view(-1)
                norms_per_layer[li].append(float(torch.linalg.norm(dv, ord=2).item()))

        # (A) true public refs if provided
        paths = self.center_clip_ref_model_paths
        if paths and len(paths) > 0:
            for pth in paths:
                try:
                    ref_model = None
                    if hasattr(type(self.model), "from_pretrained"):
                        try:
                            ref_model = type(self.model).from_pretrained(pth, torch_dtype=self.model.dtype)
                        except Exception:
                            ref_model = None
                    if ref_model is not None:
                        ref_sd = { _canon_name(k): v.detach().cpu() for (k, v) in ref_model.named_parameters() }
                        accumulate_from_state_dict(ref_sd)
                        del ref_model
                        torch.cuda.empty_cache()
                        continue
                    # common files
                    candidate_files = [
                        "pytorch_model.bin", "adapter_model.bin", "consolidated.00.pth", "model.safetensors",
                    ]
                    import os as _os
                    loaded = False
                    for fn in candidate_files:
                        fp = _os.path.join(pth, fn)
                        if _os.path.exists(fp):
                            try:
                                if fp.endswith(".safetensors"):
                                    from safetensors.torch import load_file as _load_sft
                                    sd = _load_sft(fp, device="cpu")
                                else:
                                    sd = torch.load(fp, map_location="cpu")
                                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                                    sd = sd["state_dict"]
                                ref_sd = { _canon_name(k): v for k, v in sd.items() if isinstance(v, torch.Tensor) }
                                accumulate_from_state_dict(ref_sd)
                                loaded = True
                                break
                            except Exception:
                                continue
                    if not loaded:
                        logger.warning("PUM: could not load reference model at %s for quantile C_l", pth)
                except Exception as e:
                    logger.warning("PUM: error loading reference '%s' for quantile C_l: %s", str(pth), str(e))

        # (B) synthetic public fallback
        if all(len(v) == 0 for v in norms_per_layer) and self.center_clip_ref_synth_J > 0:
            if (self.synth_mode or "gaussian") == "gaussian":
                # BOUNDED: C_ℓ ≈ κ · ρ · RMS(θ_ref,ℓ) per synthetic sample; quantile ⇒ same scale.
                rho = max(float(self.synth_gauss_rho), 0.0)
                layer_rms_ref = self._compute_layer_rms_from_sd(self._theta_ref_sd)  # CURRENT REF, not base
                if layer_rms_ref is None:
                    logger.warning("PUM: layer RMS unavailable; gaussian synth fallback cannot scale per layer.")
                J = int(self.center_clip_ref_synth_J)
                for j in range(J):
                    for li in range(L):
                        rms_i = float(layer_rms_ref[li]) if (layer_rms_ref is not None and li < len(layer_rms_ref)) else 0.0
                        norms_per_layer[li].append(rho * rms_i)
                logger.info("PUM: synthetic public refs used for C_l (mode=gaussian, rho=%.4g, J=%d).", rho, J)
            else:
                # function-preserving (old) — can be large; keep for completeness
                J = int(self.center_clip_ref_synth_J)
                for j in range(J):
                    tmp = copy.deepcopy(self.model)
                    self._materialize_params(tmp, self._theta_ref_sd)  # anchor at θ_ref (not base)
                    Tj = self._sample_T(tmp, seed=self._base_seed + 13 * (j + 1), force=True)
                    if Tj is None:
                        continue
                    self._apply_T(tmp, Tj)
                    ref_sd = self._state_dict_tensors(tmp)
                    accumulate_from_state_dict(ref_sd)
                    del tmp
                    torch.cuda.empty_cache()
                logger.info("PUM: synthetic public refs used for C_l (mode=function_preserving, J=%d).", J)

        # finalize quantiles
        q = min(max(self.center_clip_quantile_q, 0.0), 1.0)
        kappa = max(self.center_clip_quantile_kappa, 0.0)
        Cs: List[float] = []
        for li in range(L):
            vals = norms_per_layer[li]
            if not vals:
                Cs.append(0.0)
            else:
                t = torch.tensor(vals, dtype=torch.float32)
                c_l = float(torch.quantile(t, q).item()) * kappa
                Cs.append(c_l)

        # apply user caps/floors relative to RMS(θ_ref,ℓ)
        layer_rms_ref = self._compute_layer_rms_from_sd(self._theta_ref_sd)
        if layer_rms_ref is not None:
            for li in range(L):
                R_i = float(layer_rms_ref[li]) if li < len(layer_rms_ref) else 0.0
                if self.c_rel_clip is not None and self.c_rel_clip > 0 and R_i > 0:
                    Cs[li] = min(Cs[li], self.c_rel_clip * R_i)
                if self.c_rel_floor is not None and self.c_rel_floor > 0 and R_i > 0:
                    Cs[li] = max(Cs[li], self.c_rel_floor * R_i)

        self._center_clip_C_from_quantile = Cs
        if len(Cs) >= 3:
            logger.info("PUM: C_l preview [0:3]=%s ... (q=%.2f, κ=%.2f; anchored at θ_ref)", str(Cs[:3]), q, kappa)

    def _compute_center_clip_delta(
        self,
        current_sd: Dict[str, torch.Tensor],
        ref_sd: Dict[str, torch.Tensor],
        C_per_layer: Optional[List[float]],
        C_global: Optional[float],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not C_per_layer and not C_global:
            return None
        L = int(self._num_layers) if self._num_layers is not None else 0
        layer_sq = [0.0 for _ in range(max(L, 0))]
        other_sq = 0.0
        q_cache: Dict[str, torch.Tensor] = {}
        for name, cur in current_sd.items():
            ref = ref_sd.get(name, cur)
            q = (cur - ref)
            q_cache[name] = q
            li = self._param_layer_index(name)
            if C_per_layer is not None and li is not None and li < len(C_per_layer):
                layer_sq[li] += q.detach().float().pow(2).sum().item()
            else:
                other_sq += q.detach().float().pow(2).sum().item()

        scales_layer: List[float] = []
        if C_per_layer is not None and len(C_per_layer) > 0:
            for li in range(len(C_per_layer)):
                C = float(C_per_layer[li]) if C_per_layer[li] is not None else 0.0
                if C <= 0:
                    scales_layer.append(0.0)
                else:
                    nrm = math.sqrt(max(layer_sq[li], 1e-12))
                    scales_layer.append(min(1.0, C / nrm))
        s_other = None
        if C_global is not None and C_global > 0:
            nrm = math.sqrt(max(other_sq, 1e-12))
            s_other = min(1.0, C_global / nrm)

        delta: Dict[str, torch.Tensor] = {}
        for name, q in q_cache.items():
            li = self._param_layer_index(name)
            if C_per_layer is not None and li is not None and li < len(scales_layer):
                s = scales_layer[li]
                delta[name] = -q if s == 0.0 else (torch.zeros_like(q) if s == 1.0 else (s - 1.0) * q)
            elif s_other is not None:
                s = s_other
                delta[name] = -q if s == 0.0 else (torch.zeros_like(q) if s == 1.0 else (s - 1.0) * q)
            else:
                delta[name] = torch.zeros_like(q)
            # --- diagnostics for center clipping ---
            try:
                if os.environ.get("PUM_LOG_C_DETAIL", "0").strip() not in ("", "0", "false", "False"):
                    shrunk = []
                    for li, s in enumerate(scales_layer or []):
                        if s < 0.999:  # actually shrunk
                            shrunk.append(li)
                    if shrunk or (s_other is not None and s_other < 0.999):
                        import statistics as _stats
                        scales_used = [scales_layer[li] for li in shrunk] if shrunk else []
                        if s_other is not None and not (scales_used):
                            scales_used = [s_other]
                        if scales_used:
                            med = _stats.median(scales_used); mn = min(scales_used); mx = max(scales_used)
                            logger.info("PUM center-clip: shrunk %d/%d layers; scale s_l median=%.3g, min=%.3g, max=%.3g",
                                        len(shrunk), len(scales_layer or []), med, mn, mx)
                        # Print a few worst layers (smallest scales)
                        worst = sorted(shrunk, key=lambda i: scales_layer[i])[:8]
                        for li in worst:
                            nrm = math.sqrt(max(layer_sq[li], 1e-12))
                            Cval = float(C_per_layer[li]) if (C_per_layer and li < len(C_per_layer)) else float('nan')
                            logger.info("PUM center-clip detail: layer %d: ||q||_2=%.6g, C_l=%.6g, scale=%.3g",
                                        li, nrm, Cval, scales_layer[li])
                    else:
                        logger.info("PUM center-clip: no layers shrunk this round.")
            except Exception as _e:
                logger.warning("PUM center-clip diagnostics failed: %s", str(_e))
        return delta
