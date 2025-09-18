# pum.py — Perturb–Unlearn–Merge meta-trainer (public-only DP calibration, deterministic seeds)
# Implements:
#   • public-only quantile C_ell  → per-layer sensitivities Δ̄_{2,ℓ}=2C_ℓ → RDP calibration of σ_ℓ
#   • synthetic public fallback for C_ℓ when no ref models are provided
#   • identical inner-trainer seeds across copies/rounds (reproducibility)
#   • tiny jitter τ_n = 1e-4 * σ_n per parameter
#   • automatic adoption of nested Hydra `trainer.method_args.*` overrides (e.g., inner_handler)

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


PUM_FORCE_MAX_ROUNDS = int(os.environ.get("PUM_FORCE_MAX_ROUNDS", "0") or "0")
PUM_FORCE_MAX_COPIES = int(os.environ.get("PUM_FORCE_MAX_COPIES", "0") or "0")
_tmp = os.environ.get("PUM_FORCE_LOCAL_MAX_STEPS", "").strip()
PUM_FORCE_LOCAL_MAX_STEPS = int(_tmp) if _tmp.isdigit() and int(_tmp) > 0 else None


# ----------------------------
# Utilities
# ----------------------------
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


def _named_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(n, p) for (n, p) in model.named_parameters() if p is not None and p.requires_grad]


def _zero_like_param_dict(model: nn.Module, device: Optional[torch.device] = None):
    d = {}
    for n, p in _named_params(model):
        d[n] = torch.zeros_like(p, device=device or p.device)
    return d


def _state_dict_tensors(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for (n, p) in _named_params(model)}


def _apply_state_dict_delta(model: nn.Module, delta: Dict[str, torch.Tensor], scale: float = 1.0):
    with torch.no_grad():
        for n, p in _named_params(model):
            if n in delta:
                p.add_(scale * delta[n].to(p.device))


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
    # sample on CPU for generator portability, then move
    z = torch.randn(t.shape, dtype=t.dtype, device="cpu", generator=gen) * std
    return z.to(device=t.device, dtype=t.dtype)


def _generate_zero_sum_noises(
    model: nn.Module,
    m: int,
    sigma: float,
    device: Optional[torch.device] = None,
    per_param_sigma: Optional[Dict[str, float]] = None,
    rng: Optional[torch.Generator] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Generate m zero-sum, equal-variance Gaussian noises per-parameter.

    If m == 1 or all sigmas == 0, returns zeros.
    """
    if m <= 1:
        return [_zero_like_param_dict(model, device=device)]

    # Fast path: no noise at all
    if per_param_sigma is None and not (sigma and sigma > 0.0):
        return [_zero_like_param_dict(model, device=device) for _ in range(m)]

    def _std_for(n: str, p: torch.Tensor) -> float:
        if per_param_sigma is not None:
            s = float(per_param_sigma.get(n, 0.0))
            return s if s > 0 else 0.0
        return float(sigma)

    # Check if any positive std
    any_pos = False
    for n, p in _named_params(model):
        if _std_for(n, p) > 0:
            any_pos = True
            break
    if not any_pos:
        return [_zero_like_param_dict(model, device=device) for _ in range(m)]

    param_names = [n for (n, _) in _named_params(model)]
    noises = [{n: None for n in param_names} for _ in range(m)]
    for n, p in _named_params(model):
        sig_n = _std_for(n, p)
        if sig_n <= 0:
            for k in range(m):
                noises[k][n] = torch.zeros_like(p, device=device or p.device)
            continue
        zs = [_randn_like_with_generator(p.to(device or p.device), sig_n, rng) for _ in range(m)]
        z_mean = torch.stack(zs, dim=0).mean(dim=0)
        scale = math.sqrt(m / (m - 1))
        for k in range(m):
            noises[k][n] = (zs[k] - z_mean) * scale
    return noises


# ----------------------------
# PUM Trainer
# ----------------------------
class PUM(FinetuneTrainer):
    """
    Perturb–Unlearn–Merge (PUM) meta-trainer
      – public-only quantile C_ell → per-layer sensitivities → DP-calibrated σ_ell
      – deterministic seeds (identical across copies/rounds)
      – τ_n = 1e-4 σ_n jitter
      – accepts nested Hydra `trainer.method_args.*` overrides
    """

    def __init__(
        self,
        # PUM-specific args (top-level)
        inner_handler: str = "GradAscent",
        inner_method_args: Optional[dict] = None,
        copies_m: int = 4,
        rounds_R: int = 1,
        sigma: float = 0.0,
        sigma_per_layer: Optional[List[float]] = None,
        per_layer_noise: bool = False,
        # DP calibration
        dp_epsilon: Optional[float] = None,
        dp_delta: Optional[float] = None,
        dp_sensitivity_total_l2: Optional[float] = None,
        dp_sensitivity_per_layer_l2: Optional[List[float]] = None,
        dp_rdp_orders: Optional[List[float]] = None,
        dp_use_worstcase_alpha: bool = True,
        dp_per_layer_allocation: str = "auto",  # "auto" | "equalized" | "varmin"
        alpha_min: float = 1.0,
        alpha_max: float = 1.1,
        eta_srv: float = 1.0,
        # server-side clipping center + reference
        theta_ref_beta: float = 0.8,
        server_center_clipping: Optional[bool] = None,
        center_clip_C_global: Optional[float] = None,
        center_clip_C_per_layer: Optional[List[float]] = None,
        center_clip_quantile_q: float = 0.95,
        center_clip_quantile_kappa: float = 1.25,
        center_clip_round_gamma: float = 1.3,
        center_clip_ref_model_paths: Optional[List[str]] = None,
        # jitter control (we will use τ_n = jitter_rel_to_sigma * σ_n; the absolute jitter_tau is ignored if rel>0)
        jitter_tau: float = 0.0,
        jitter_rel_to_sigma: float = 1e-4,
        # Local budget
        local_epochs: int = 1,
        local_max_steps: Optional[int] = None,
        auto_balance_local_max_steps: bool = True,
        # Server-side update clipping (safeguard)
        clip_update_norm: Optional[float] = None,
        clip_update_norm_per_layer: Optional[List[float]] = None,
        # Reparam
        use_orthogonal_reparam: bool = False,
        # Synthetic public fallback for C_ℓ if no ref paths are provided
        center_clip_ref_synth_J: int = 8,
        # base trainer args
        *args,
        **kwargs,
    ):
        # --- absorb nested Hydra method_args (if present) before calling super()
        # This lets CLI overrides like `trainer.method_args.inner_handler=NPO` take effect.
        method_args_override = kwargs.pop("method_args", None)

        super().__init__(*args, **kwargs)

        def _as_bool(x, default=False):
            if isinstance(x, bool) or x is None:
                return bool(x) if x is not None else default
            if isinstance(x, str):
                s = x.strip().lower()
                if s in ("true", "1", "yes", "y"): return True
                if s in ("false", "0", "no", "n", "null", "none", ""): return False
            return bool(x)

        def _as_list_of_numbers(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return [float(v) for v in x]
            if isinstance(x, str):
                s = x.strip()
                if s.lower() in ("null", "none", ""):
                    return None
                import ast
                try:
                    val = ast.literal_eval(s)
                    if isinstance(val, (list, tuple)):
                        return [float(v) for v in val]
                except Exception:
                    pass
                try:
                    return [float(v) for v in s.split(",") if v.strip()]
                except Exception:
                    return None
            try:
                return [float(x)]
            except Exception:
                return None

        def _as_list_of_strings(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return [str(v) for v in x]
            if isinstance(x, str):
                s = x.strip()
                if s.lower() in ("null", "none", ""):
                    return None
                import ast
                try:
                    val = ast.literal_eval(s)
                    if isinstance(val, (list, tuple)):
                        return [str(v) for v in val]
                except Exception:
                    pass
                if "," in s:
                    return [u.strip() for u in s.split(",") if u.strip()]
                return [s]
            return [str(x)]

        # assign fields
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
        self.dp_rdp_orders = _as_list_of_numbers(dp_rdp_orders)
        self.dp_use_worstcase_alpha = _as_bool(dp_use_worstcase_alpha, default=True)
        self.dp_per_layer_allocation = str(dp_per_layer_allocation).lower().strip()
        if self.dp_per_layer_allocation not in ("auto", "equalized", "varmin"):
            logger.warning("PUM: dp_per_layer_allocation '%s' not in {auto,equalized,varmin}; defaulting to 'auto'",
                           self.dp_per_layer_allocation)
            self.dp_per_layer_allocation = "auto"
        self.alpha_min = max(float(alpha_min), 1.0)
        self.alpha_max = max(float(alpha_max), self.alpha_min)
        self.eta_srv = float(eta_srv)
        self.theta_ref_beta = float(theta_ref_beta)

        self.center_clip_C_global = float(center_clip_C_global) if center_clip_C_global is not None else None
        self.center_clip_C_per_layer = list(center_clip_C_per_layer) if center_clip_C_per_layer is not None else None
        self.center_clip_quantile_q = float(center_clip_quantile_q)
        self.center_clip_quantile_kappa = float(center_clip_quantile_kappa)
        self.center_clip_round_gamma = float(center_clip_round_gamma)
        self.center_clip_ref_model_paths = _as_list_of_strings(center_clip_ref_model_paths)

        self.jitter_tau = float(jitter_tau)
        self.jitter_rel_to_sigma = float(jitter_rel_to_sigma)

        self.local_epochs = int(local_epochs)
        self.local_max_steps = int(local_max_steps) if local_max_steps is not None else None
        self.auto_balance_local_max_steps = _as_bool(auto_balance_local_max_steps, default=True)
        self.clip_update_norm = clip_update_norm
        self.clip_update_norm_per_layer = list(clip_update_norm_per_layer) if clip_update_norm_per_layer is not None else None
        self.use_orthogonal_reparam = _as_bool(use_orthogonal_reparam, default=False)

        self.center_clip_ref_synth_J = max(0, int(center_clip_ref_synth_J))

        if self.copies_m < 1:
            raise ValueError("copies_m must be >= 1")

        # absorb nested overrides if Hydra passed `trainer.method_args.*`
        if isinstance(method_args_override, dict):
            _known = {
                "inner_handler", "inner_method_args", "copies_m", "rounds_R",
                "sigma", "sigma_per_layer", "per_layer_noise",
                "dp_epsilon", "dp_delta", "dp_sensitivity_total_l2", "dp_sensitivity_per_layer_l2",
                "dp_rdp_orders", "dp_use_worstcase_alpha", "dp_per_layer_allocation",
                "alpha_min", "alpha_max", "eta_srv",
                "theta_ref_beta", "server_center_clipping",
                "center_clip_C_global", "center_clip_C_per_layer", "center_clip_quantile_q",
                "center_clip_quantile_kappa", "center_clip_round_gamma", "center_clip_ref_model_paths",
                "jitter_tau", "jitter_rel_to_sigma",
                "local_epochs", "local_max_steps", "auto_balance_local_max_steps",
                "clip_update_norm", "clip_update_norm_per_layer",
                "use_orthogonal_reparam",
                "center_clip_ref_synth_J",
            }
            for k, v in method_args_override.items():
                if k in _known:
                    setattr(self, k, v if k not in {"sigma_per_layer", "dp_rdp_orders", "center_clip_C_per_layer",
                                                    "dp_sensitivity_per_layer_l2", "center_clip_ref_model_paths"} else
                            (v if isinstance(v, list) else (v if v is None else [v])))
                elif k == "inner":
                    self.inner_handler = str(v)
                elif k == "inner_method_args" and isinstance(v, dict):
                    self.inner_method_args.update(v)
                else:
                    # ignore unknown keys silently
                    pass

        # Apply quick-test caps
        if PUM_FORCE_MAX_ROUNDS and PUM_FORCE_MAX_ROUNDS > 0:
            self.rounds_R = min(self.rounds_R, PUM_FORCE_MAX_ROUNDS)
        if PUM_FORCE_MAX_COPIES and PUM_FORCE_MAX_COPIES > 0:
            self.copies_m = min(self.copies_m, PUM_FORCE_MAX_COPIES)
        if PUM_FORCE_LOCAL_MAX_STEPS is not None and PUM_FORCE_LOCAL_MAX_STEPS > 0:
            self.local_max_steps = PUM_FORCE_LOCAL_MAX_STEPS
            self.auto_balance_local_max_steps = False

        # seed & model shape meta
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

        # Enable server clipping by default if any thresholds/sensitivities are defined or ref models provided
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

        # EMA reference & caches
        self._theta_ref_sd: Dict[str, torch.Tensor] = _state_dict_tensors(self.model)
        self._theta_base_sd: Dict[str, torch.Tensor] = {k: v.clone().detach().cpu() for k, v in self._theta_ref_sd.items()}
        denom = sum(v.numel() for v in self._theta_base_sd.values())
        if denom <= 0:
            self._base_global_rms = 0.0
        else:
            num = sum(v.detach().float().pow(2).sum().item() for v in self._theta_base_sd.values())
            self._base_global_rms = math.sqrt(max(num / max(denom, 1), 0.0))
        self._pub_mean_prev_sd: Optional[Dict[str, torch.Tensor]] = None
        self._center_clip_C_from_quantile: Optional[List[float]] = None

        # Guardrail: if everything is zero AND no public reference to calibrate from
        if (not self.dp_epsilon or not self.dp_delta) and (self.sigma <= 0.0) and (self.sigma_per_layer is None) and \
           (self.center_clip_ref_model_paths in (None, []) and self.center_clip_ref_synth_J <= 0):
            logger.warning(
                "PUM: No DP targets and no reference models for public-only calibration; sigma==0. "
                "All copies may become identical. Provide public refs or enable synthetic refs (center_clip_ref_synth_J>0)."
            )

    # ----------------------------
    # DP calibration for sigma
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
            return [1.5, 2, 3, 4, 8, 16, 32, 64, 128]
        orders = [float(x) for x in self.dp_rdp_orders if float(x) > 1.0]
        return orders or [2.0, 4.0, 8.0, 16.0]

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

    def _calibrate_sigma_from_dp(self) -> None:
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

        # per-layer
        if self.per_layer_noise:
            L = int(self._num_layers) if self._num_layers is not None else None
            if not L or L <= 0:
                logger.warning("PUM DP per-layer calibration requested but model layers not found; falling back to single-sigma.")
            else:
                Delta_l = self._get_per_layer_sens(L)
                if Delta_l is None:
                    logger.warning("PUM DP per-layer calibration requires per-layer/total sensitivity; falling back to single-sigma.")
                else:
                    best_sigmas: Optional[List[float]] = None
                    best_total_var: Optional[float] = None
                    for lam in orders:
                        A = eps_tgt - (math.log(1.0 / delta) / (lam - 1.0))
                        if A <= 0:
                            continue
                        K = (2.0 * A) / (R * lam * max(S_alpha, 1e-12))
                        if K <= 0:
                            continue
                        # equalized: σ_l = κ Δ_l, κ = sqrt(L/K)
                        kappa = math.sqrt(float(L) / K)
                        sig_eq = [kappa * d for d in Delta_l]
                        tot_eq = (kappa * kappa) * sum(d * d for d in Delta_l)
                        # var-min: σ_l = c sqrt(Δ_l), c = sqrt((sum Δ_l)/K)
                        sum_D = sum(Delta_l)
                        c = math.sqrt(max(sum_D, 0.0) / K)
                        sig_vm = [c * math.sqrt(max(d, 0.0)) for d in Delta_l]
                        tot_vm = (c * c) * sum_D
                        # choose smaller total variance
                        for total_var, sigmas in ((tot_eq, sig_eq), (tot_vm, sig_vm)):
                            if not all(math.isfinite(s) and s >= 0 for s in sigmas):
                                continue
                            if (best_total_var is None) or (total_var < best_total_var):
                                best_total_var = total_var
                                best_sigmas = sigmas
                    if best_sigmas is not None:
                        self.sigma_per_layer = [float(s) for s in best_sigmas]
                        self.sigma = 0.0  # avoid adding noise to non-layer params by default
                        calibrated = True
                        logger.info("PUM DP per-layer calibration: σ[0:3]=%s ...", str(self.sigma_per_layer[:3]))

        if not calibrated:
            # single-sigma (requires total sensitivity)
            if self.dp_sens_tot_l2 is None:
                self._apply_sigma_safety_clip()
                return
            Delta2_tot = float(self.dp_sens_tot_l2)
            best_sigma = None
            for lam in orders:
                A = eps_tgt - (math.log(1.0 / delta) / (lam - 1.0))
                if A <= 0:
                    continue
                sig = Delta2_tot * math.sqrt((R * lam * max(S_alpha, 1e-12)) / (2.0 * A))
                if sig <= 0 or math.isnan(sig) or math.isinf(sig):
                    continue
                if (best_sigma is None) or (sig < best_sigma):
                    best_sigma = sig
            if best_sigma is None:
                logger.warning("PUM DP-calibration failed; keeping existing sigma(s).")
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
            gamma = self.center_clip_round_gamma if (self.rounds_R and self.rounds_R > 1) else 1.0
            C_per_layer = [gamma * float(c) for c in self._center_clip_C_from_quantile]
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

    def _materialize_params(self, model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in _named_params(model):
                v = state.get(n)
                if v is not None:
                    p.copy_(v.to(device=p.device, dtype=p.dtype))

    def _sample_T(self, model: nn.Module, seed: int, force: bool = False):
        """Sample per-layer transforms (orthogonals for attn head_dim; permutations for FFN).
        If force=True, ignore self.use_orthogonal_reparam gate (used for synthetic refs)."""
        if not force and not self.use_orthogonal_reparam:
            return None
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
            if attn is not None:
                head_dim = int(getattr(attn, "head_dim", 0) or (attn.q_proj.weight.shape[0] //
                                                                getattr(attn, "num_heads", 1)))
                if head_dim > 0:
                    A = torch.randn(head_dim, head_dim, device="cpu", dtype=torch.float32)
                    Q, _ = torch.linalg.qr(A)
                    entry["R_sub"] = Q.detach().to(device=attn.q_proj.weight.device,
                                                   dtype=attn.q_proj.weight.dtype)
                    entry["n_q_heads"] = torch.tensor(int(getattr(attn, "num_heads", 0)), device="cpu")
                    entry["n_kv_heads"] = torch.tensor(int(getattr(attn, "num_key_value_heads",
                                                                   getattr(attn, "num_heads", 0))), device="cpu")
                    entry["head_dim"] = torch.tensor(head_dim, device="cpu")
            if mlp is not None:
                inter = int(mlp.up_proj.weight.shape[0])
                perm = torch.randperm(inter, device=mlp.up_proj.weight.device)
                inv_perm = torch.empty_like(perm)
                inv_perm[perm] = torch.arange(inter, device=perm.device)
                entry["perm"] = perm
                entry["inv_perm"] = inv_perm
            if entry:
                T[li] = entry
        return T if len(T) > 0 else None

    def _apply_T(self, model: nn.Module, T) -> None:
        if not T:
            return
        try:
            layers = model.model.layers  # type: ignore[attr-defined]
        except Exception:
            return
        for li, layer in enumerate(layers):
            if li not in T:
                continue
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
        if not T:
            return update
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
    # Helpers
    # ----------------------------
    def _apply_sigma_safety_clip(self) -> None:
        if not hasattr(self, "_base_global_rms"):
            return
        try:
            sigma_rel_clip = float(os.environ.get("PUM_SIGMA_REL_CLIP", "0.25"))
        except (TypeError, ValueError):
            sigma_rel_clip = 0.25
        sigma_rel_clip = max(sigma_rel_clip, 0.0)
        base_rms = max(getattr(self, "_base_global_rms", 0.0), 0.0)
        if sigma_rel_clip <= 0.0 or base_rms <= 0.0:
            return
        limit = sigma_rel_clip * max(base_rms, 1e-8)
        clamped = False
        if self.sigma_per_layer is not None:
            new_sigmas: List[float] = []
            for s in self.sigma_per_layer:
                try:
                    s_val = float(s) if s is not None else 0.0
                except (TypeError, ValueError):
                    s_val = 0.0
                s_val = max(s_val, 0.0)
                if s_val > limit:
                    s_val = limit
                    clamped = True
                new_sigmas.append(s_val)
            self.sigma_per_layer = new_sigmas
        if self.sigma is not None:
            try:
                sigma_scalar = float(self.sigma)
            except (TypeError, ValueError):
                sigma_scalar = 0.0
            sigma_scalar = max(sigma_scalar, 0.0)
            if sigma_scalar > limit:
                sigma_scalar = limit
                clamped = True
            self.sigma = sigma_scalar
        if clamped:
            logger.info(
                "PUM: sigma clipped to <= %.6g (rel clip=%.3g, base_rms=%.6g)",
                limit,
                sigma_rel_clip,
                base_rms,
            )

    def _build_per_param_sigma(self) -> Optional[Dict[str, float]]:
        if self.sigma_per_layer is None:
            return None
        sigma_other = 0.0 if self.per_layer_noise else (self.sigma if (self.sigma and self.sigma > 0) else 0.0)
        per_param: Dict[str, float] = {}
        for n, _ in _named_params(self.model):
            li = self._param_layer_index(n)
            if li is not None and li < len(self.sigma_per_layer):
                per_param[n] = float(self.sigma_per_layer[li])
            else:
                per_param[n] = float(sigma_other)
        return per_param

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

    def _maybe_init_center_C_from_quantile(self) -> None:
        if self._center_clip_C_from_quantile is not None:
            return
        if not self._num_layers or self._num_layers <= 0:
            logger.warning("PUM: cannot compute quantile C_l; model layers unknown")
            return

        L = int(self._num_layers)
        norms_per_layer: List[List[float]] = [[] for _ in range(L)]

        # helper
        def accumulate_from_state_dict(sd: Dict[str, torch.Tensor]):
            for name, base_v in self._theta_base_sd.items():
                li = self._param_layer_index(name)
                if li is None or li >= L:
                    continue
                ref_v = sd.get(name, None)
                if ref_v is None:
                    continue
                dv = (ref_v.detach().cpu() - base_v).float().view(-1)
                norms_per_layer[li].append(float(torch.linalg.norm(dv, ord=2).item()))

        # (A) true public refs if provided (best)
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
                        ref_sd = {k: v.detach().cpu() for (k, v) in ref_model.named_parameters()}
                        accumulate_from_state_dict(ref_sd)
                        del ref_model
                        torch.cuda.empty_cache()
                        continue
                    # fallback to common files
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
                                ref_sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
                                accumulate_from_state_dict(ref_sd)
                                loaded = True
                                break
                            except Exception:
                                continue
                    if not loaded:
                        logger.warning("PUM: could not load reference model at %s for quantile C_l", pth)
                except Exception as e:
                    logger.warning("PUM: error loading reference '%s' for quantile C_l: %s", str(pth), str(e))

        # (B) synthetic public fallback: apply J function-preserving transforms to θ_base
        if all(len(v) == 0 for v in norms_per_layer) and self.center_clip_ref_synth_J > 0:
            J = int(self.center_clip_ref_synth_J)
            for j in range(J):
                tmp = copy.deepcopy(self.model)
                # materialize θ_base
                self._materialize_params(tmp, self._theta_base_sd)
                # sample T on θ_base (force=True), apply, measure distance to θ_base
                Tj = self._sample_T(tmp, seed=self._base_seed + 13 * (j + 1), force=True)
                if Tj is None:
                    continue
                self._apply_T(tmp, Tj)
                ref_sd = _state_dict_tensors(tmp)
                accumulate_from_state_dict(ref_sd)
                del tmp
                torch.cuda.empty_cache()
            if any(len(v) > 0 for v in norms_per_layer):
                logger.info("PUM: synthetic public refs used for C_l (J=%d).", J)
            else:
                logger.warning("PUM: synthetic public refs failed; C_l may remain zero without manual sensitivities.")

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
        self._center_clip_C_from_quantile = Cs

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
        return delta

    def _clip_update(self, update: Dict[str, torch.Tensor]):
        if self.clip_update_norm_per_layer is not None and self._num_layers:
            L = min(int(self._num_layers), len(self.clip_update_norm_per_layer))
            layer_sq = [0.0 for _ in range(L)]
            for name, v in update.items():
                li = self._param_layer_index(name)
                if li is None or li >= L:
                    continue
                layer_sq[li] += v.detach().float().pow(2).sum().item()
            scales = [1.0 for _ in range(L)]
            for li in range(L):
                thr = float(self.clip_update_norm_per_layer[li])
                if thr is None or thr <= 0:
                    continue
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
    # Main train loop
    # ----------------------------
    def _make_inner_args(self, output_dir: str) -> TrainingArguments:
        """Clone outer args; deterministic identical seed across copies/rounds."""
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
            try:
                inner_args.bf16 = False
            except Exception:
                pass
            try:
                inner_args.fp16 = False
            except Exception:
                pass
            try:
                inner_args.torch_compile = False
            except Exception:
                pass
        # identical seed for all copies/rounds (reproducible; diversity from nonzero noise)
        inner_args.seed = (self._base_seed + 17) % (2**31 - 1)
        try:
            inner_args.data_seed = inner_args.seed
        except Exception:
            pass
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

    def train(self, resume_from_checkpoint: Optional[str] = None, **kwargs):
        self.model.train()

        logger.info(
            "PUM config: inner=%s, m=%d, R=%d, per_layer_noise=%s, σ=%s, has_σ_per_layer=%s, server_clip=%s",
            self.inner_handler, self.copies_m, self.rounds_R, str(self.per_layer_noise),
            f"{self.sigma:.6g}", str(self.sigma_per_layer is not None), str(self.server_center_clipping)
        )

        self._maybe_set_auto_local_max_steps()

        # Compute public-only C_l (ref models or synthetic) BEFORE DP calibration
        if self.server_center_clipping:
            self._maybe_init_center_C_from_quantile()

        # Now calibrate σ from DP targets (uses Δ̄_{2,ℓ}=2C_ℓ if available)
        self._calibrate_sigma_from_dp()

        for r in range(1, self.rounds_R + 1):
            # EMA ref from previous round's published mean
            if self._pub_mean_prev_sd is not None:
                beta = float(self.theta_ref_beta)
                with torch.no_grad():
                    for n, _ in _named_params(self.model):
                        prev_ref = self._theta_ref_sd.get(n)
                        pub = self._pub_mean_prev_sd.get(n, prev_ref)
                        if prev_ref is None:
                            self._theta_ref_sd[n] = pub.detach().clone()
                        else:
                            self._theta_ref_sd[n] = (1.0 - beta) * prev_ref + beta * pub

            # Re-run DP calibration to pick updated C_l (if any) and S_alpha
            self._calibrate_sigma_from_dp()

            # Seeded RNG for base noises (deterministic per round)
            rng = torch.Generator(device="cpu")
            rng.manual_seed(self._base_seed + 811 * r)

            per_param_sigma = self._build_per_param_sigma()
            base_noises = _generate_zero_sum_noises(
                self.model, self.copies_m, self.sigma, per_param_sigma=per_param_sigma, rng=rng
            )

            # secret scales α_k ~ Uniform[α_min, α_max] (deterministic per round)
            alphas: List[float] = []
            g = torch.Generator(device="cpu")
            g.manual_seed(self._base_seed + 9973 * r)
            for _ in range(self.copies_m):
                if self.alpha_max == self.alpha_min:
                    alphas.append(self.alpha_min)
                else:
                    u = torch.rand(1, generator=g).item()
                    alphas.append(self.alpha_min + u * (self.alpha_max - self.alpha_min))

            # streaming accumulators
            S0 = 0.0
            S1 = _zero_like_param_dict(self.model)
            pub_sum_server = _zero_like_param_dict(self.model)

            # for per-round quantile update using aligned published models
            round_norms_per_layer: Optional[List[List[float]]] = (
                [[] for _ in range(int(self._num_layers))]
                if (self._num_layers is not None and int(self._num_layers) > 0)
                else None
            )

            cur_sd = _state_dict_tensors(self.model)
            center_delta = None
            if self.server_center_clipping:
                C_per_layer, C_global = self._resolve_center_clip_thresholds()
                center_delta = self._compute_center_clip_delta(cur_sd, self._theta_ref_sd, C_per_layer, C_global)

            for k in range(self.copies_m):
                alpha_k = float(alphas[k])

                # correlated perturbation for this copy
                eps_k = {n: alpha_k * base_noises[k][n] for (n, _) in _named_params(self.model)}

                # tiny jitter per parameter: τ_n = 1e-4 * σ_n
                xi_k = None
                # pick σ_n for each parameter
                if self.jitter_rel_to_sigma and self.jitter_rel_to_sigma > 0.0:
                    xi_k = {}
                    for n, p in _named_params(self.model):
                        if per_param_sigma is not None:
                            sig_n = float(per_param_sigma.get(n, 0.0))
                        else:
                            sig_n = float(self.sigma or 0.0)
                        tau_n = float(self.jitter_rel_to_sigma) * sig_n
                        if tau_n > 0:
                            xi_k[n] = _randn_like_with_generator(p, tau_n, rng)
                        else:
                            xi_k[n] = torch.zeros_like(p)

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

                before_sd = _state_dict_tensors(inner_trainer.model)
                before_sd_aligned = self._apply_T_inv_to_update(before_sd, T_k)
                with torch.no_grad():
                    for n, _ in _named_params(self.model):
                        clean = torch.nan_to_num(
                            before_sd_aligned[n], nan=0.0, posinf=0.0, neginf=0.0
                        )
                        pub_sum_server[n].add_(clean.to(pub_sum_server[n].device))
                if round_norms_per_layer is not None:
                    for name, base_v in self._theta_base_sd.items():
                        li = self._param_layer_index(name)
                        if li is None or li >= len(round_norms_per_layer):
                            continue
                        v = before_sd_aligned.get(name, None)
                        if v is None:
                            continue
                        dv = (v.detach().cpu() - base_v).float().view(-1)
                        nrm = float(torch.linalg.norm(dv, ord=2).item())
                        round_norms_per_layer[li].append(nrm)

                inner_trainer.train()

                after_sd = _state_dict_tensors(inner_trainer.model)
                delta_k = _diff_state_dict(after_sd, before_sd)
                delta_k = self._apply_T_inv_to_update(delta_k, T_k)

                delta_k = self._clip_update(delta_k)

                bad = False
                for name, tensor in delta_k.items():
                    cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                    if not torch.isfinite(cleaned).all():
                        bad = True
                        break
                    delta_k[name] = cleaned
                if bad:
                    logger.warning("PUM: skipping copy %d (round %d) due to non-finite delta.", k + 1, r)
                    del inner_trainer, model_k, after_sd, delta_k
                    torch.cuda.empty_cache()
                    continue

                w = 1.0 / max(alpha_k, 1e-12)
                S0 += w
                for n, dv in delta_k.items():
                    if n in S1:
                        S1[n].add_(w * dv.to(S1[n].device))

                del inner_trainer, model_k, after_sd, delta_k
                torch.cuda.empty_cache()

            inv_S0 = 1.0 / max(S0, 1e-12)
            with torch.no_grad():
                total_sq = 0.0
                for n, p in _named_params(self.model):
                    update = torch.nan_to_num(S1[n] * inv_S0, nan=0.0, posinf=0.0, neginf=0.0).to(p.device)
                    total_sq += update.detach().float().pow(2).sum().item()
                    p.add_(self.eta_srv * update)
                logger.info("[PUM][round %d] ||mean update||_2 = %.6f", r, math.sqrt(max(total_sq, 0.0)))

            m = float(self.copies_m)
            self._pub_mean_prev_sd = {n: (pub_sum_server[n] / m).detach().clone() for n in pub_sum_server}

            if round_norms_per_layer is not None:
                q = min(max(self.center_clip_quantile_q, 0.0), 1.0)
                kappa = max(self.center_clip_quantile_kappa, 0.0)
                gamma = self.center_clip_round_gamma if (self.rounds_R and self.rounds_R > 1) else 1.0
                Cs: List[float] = []
                for li in range(len(round_norms_per_layer)):
                    vals = round_norms_per_layer[li]
                    if not vals:
                        Cs.append(0.0)
                    else:
                        t = torch.tensor(vals, dtype=torch.float32)
                        c_l = float(torch.quantile(t, q).item()) * kappa * gamma
                        Cs.append(c_l)
                self._center_clip_C_from_quantile = Cs

        self.model.eval()
        try:
            self.model.save_pretrained(str(self.args.output_dir))
            if getattr(self, "tokenizer", None) is not None:
                self.tokenizer.save_pretrained(str(self.args.output_dir))
        except Exception as e:
            logger.warning("PUM: failed to save_pretrained: %s", e)
        return None
