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


def _resolve_inner_trainer(handler_name: str):
    """Resolve inner trainer class by handler name without relying on trainer.__init__ registry.

    This avoids circular imports since trainer.__init__ also imports this module.
    """
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
    return [
        (n, p)
        for (n, p) in model.named_parameters()
        if p is not None and p.requires_grad
    ]


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
    # Robust difference: intersect keys and match shapes/dtypes
    out: Dict[str, torch.Tensor] = {}
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            continue
        if va.shape != vb.shape:
            continue
        try:
            out[k] = va - vb
        except Exception:
            continue
    return out


def _generate_zero_sum_noises(
    model: nn.Module,
    m: int,
    sigma: float,
    device: Optional[torch.device] = None,
    per_param_sigma: Optional[Dict[str, float]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Generate m zero-sum, equal-variance Gaussian noises per-parameter.

    Returns a list of length m where each element is a param-name->tensor map.
    If m == 1 or sigma == 0, returns zeros.
    """
    if m <= 1 or (sigma <= 0 and not per_param_sigma):
        return [_zero_like_param_dict(model, device=device) for _ in range(m)]

    # For each parameter, construct z_k, subtract mean, scale by sqrt(m/(m-1))
    param_names = [n for (n, _) in _named_params(model)]
    noises = [{n: None for n in param_names} for _ in range(m)]

    for n, p in _named_params(model):
        # Resolve std for this parameter: prefer per_param_sigma, else global sigma
        sig_n = float(per_param_sigma.get(n, sigma)) if per_param_sigma else float(sigma)
        if sig_n <= 0:
            # No noise for this param; keep zeros
            for k in range(m):
                noises[k][n] = torch.zeros_like(p, device=device or p.device)
            continue
        zs = [torch.randn_like(p, device=device or p.device) * sig_n for _ in range(m)]
        z_mean = torch.stack(zs, dim=0).mean(dim=0)
        scale = math.sqrt(m / (m - 1))
        for k in range(m):
            noises[k][n] = (zs[k] - z_mean) * scale
    return noises


class PUM(FinetuneTrainer):
    """
    Perturb-Unlearn-Merge (PUM) meta-trainer.

    Orchestrates m locally unlearned copies per round with zero-sum correlated
    perturbations and harmonic-normalized aggregation. The inner unlearning
    routine is any existing Trainer handler (e.g., GradAscent/GradDiff/NPO/DPO).
    """

    def __init__(
        self,
        # PUM-specific args
        inner_handler: str = "GradAscent",
        inner_method_args: Optional[dict] = None,
        copies_m: int = 4,
        rounds_R: int = 1,
        sigma: float = 0.0,
        sigma_per_layer: Optional[List[float]] = None,
        # Switch: use per-layer noise (DP-calibrated) vs global noise
        per_layer_noise: bool = False,
        # DP-based noise calibration (single-sigma across layers)
        dp_epsilon: Optional[float] = None,
        dp_delta: Optional[float] = None,
        dp_sensitivity_total_l2: Optional[float] = None,
        dp_sensitivity_per_layer_l2: Optional[List[float]] = None,
        dp_rdp_orders: Optional[List[float]] = None,
        dp_use_worstcase_alpha: bool = True,
        dp_per_layer_allocation: str = "auto",  # "auto", "equalized", or "varmin"
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        eta_srv: float = 1.0,
        # DP clipping reference and per-copy jitter
        theta_ref_beta: float = 0.8,
        server_center_clipping: Optional[bool] = None,
        center_clip_C_global: Optional[float] = None,
        center_clip_C_per_layer: Optional[List[float]] = None,
        # Public-only quantile-based center clipping
        center_clip_quantile_q: float = 0.95,
        center_clip_quantile_kappa: float = 1.25,
        center_clip_round_gamma: float = 1.3,
        center_clip_ref_model_paths: Optional[List[str]] = None,
        jitter_tau: float = 0.0,
        local_epochs: int = 1,
        local_max_steps: Optional[int] = None,
        auto_balance_local_max_steps: bool = False,
        clip_update_norm: Optional[float] = None,
        clip_update_norm_per_layer: Optional[List[float]] = None,
        use_orthogonal_reparam: bool = False,  # placeholder (identity by default)
        # base trainer args
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.inner_handler = inner_handler
        self.inner_method_args = inner_method_args or {}
        self.copies_m = int(copies_m)
        self.rounds_R = int(rounds_R)
        self.sigma = float(sigma)
        self.sigma_per_layer = list(sigma_per_layer) if sigma_per_layer is not None else None
        # Robust bool parsing for Hydra CLI overrides (which may come as strings)
        def _as_bool(x, default=False):
            if isinstance(x, bool) or x is None:
                return bool(x) if x is not None else default
            if isinstance(x, str):
                s = x.strip().lower()
                if s in ("true", "1", "yes", "y"): return True
                if s in ("false", "0", "no", "n", "null", "none", ""): return False
            return bool(x)

        self.per_layer_noise = _as_bool(per_layer_noise, default=False)
        # DP args
        self.dp_epsilon = float(dp_epsilon) if dp_epsilon is not None else None
        self.dp_delta = float(dp_delta) if dp_delta is not None else None
        self.dp_sens_tot_l2 = (
            float(dp_sensitivity_total_l2) if dp_sensitivity_total_l2 is not None else None
        )
        self.dp_sens_per_layer_l2 = (
            list(dp_sensitivity_per_layer_l2) if dp_sensitivity_per_layer_l2 is not None else None
        )
        # Parse orders which may arrive as YAML list or CLI string like "[1.5,2,4]"
        def _as_list_of_numbers(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return [float(v) for v in x]
            if isinstance(x, str):
                if x.strip().lower() in ("null", "none", ""):
                    return None
                import ast
                try:
                    val = ast.literal_eval(x)
                    if isinstance(val, (list, tuple)):
                        return [float(v) for v in val]
                except Exception:
                    pass
                # fallback: comma-separated
                try:
                    return [float(v) for v in x.split(',') if v.strip()]
                except Exception:
                    return None
            # last resort
            try:
                return [float(x)]
            except Exception:
                return None

        self.dp_rdp_orders = _as_list_of_numbers(dp_rdp_orders)
        self.dp_use_worstcase_alpha = _as_bool(dp_use_worstcase_alpha, default=True)
        self.dp_per_layer_allocation = str(dp_per_layer_allocation).lower().strip()
        if self.dp_per_layer_allocation not in ("auto", "equalized", "varmin"):
            logger.warning(
                "PUM: dp_per_layer_allocation '%s' not in {auto,equalized,varmin}; defaulting to 'auto'",
                self.dp_per_layer_allocation,
            )
            self.dp_per_layer_allocation = "auto"
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.eta_srv = float(eta_srv)
        self.theta_ref_beta = float(theta_ref_beta)
        self.center_clip_C_global = (
            float(center_clip_C_global) if center_clip_C_global is not None else None
        )
        self.center_clip_C_per_layer = (
            list(center_clip_C_per_layer) if center_clip_C_per_layer is not None else None
        )
        self.center_clip_quantile_q = float(center_clip_quantile_q)
        self.center_clip_quantile_kappa = float(center_clip_quantile_kappa)
        self.center_clip_round_gamma = float(center_clip_round_gamma)
        # Parse ref model paths: YAML list or CLI string list
        def _as_list_of_strings(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return [str(v) for v in x]
            if isinstance(x, str):
                if x.strip().lower() in ("null", "none", ""):
                    return None
                import ast
                try:
                    val = ast.literal_eval(x)
                    if isinstance(val, (list, tuple)):
                        return [str(v) for v in val]
                except Exception:
                    pass
                if "," in x:
                    return [s.strip() for s in x.split(',') if s.strip()]
                return [x.strip()]
            return [str(x)]

        self.center_clip_ref_model_paths = _as_list_of_strings(center_clip_ref_model_paths)
        self.jitter_tau = float(jitter_tau)
        self.local_epochs = int(local_epochs)
        self.local_max_steps = (
            int(local_max_steps) if local_max_steps is not None else None
        )
        self.auto_balance_local_max_steps = _as_bool(auto_balance_local_max_steps, default=True)
        self.clip_update_norm = clip_update_norm
        self.clip_update_norm_per_layer = (
            list(clip_update_norm_per_layer) if clip_update_norm_per_layer is not None else None
        )
        self.use_orthogonal_reparam = _as_bool(use_orthogonal_reparam, default=False)

        if self.alpha_min < 1.0:
            self.alpha_min = 1.0
        if self.alpha_max < self.alpha_min:
            self.alpha_max = self.alpha_min

        if self.copies_m < 1:
            raise ValueError("copies_m must be >= 1")

        # Seed per-round/copy deterministically for reproducibility across ranks
        self._base_seed = int(self.args.seed or 0)

        # Introspect model depth for per-layer features
        try:
            self._num_layers = int(len(self.model.model.layers))  # type: ignore[attr-defined]
        except Exception:
            self._num_layers = None

        if self.sigma_per_layer is not None and self._num_layers is not None:
            if len(self.sigma_per_layer) != self._num_layers:
                logger.warning(
                    "PUM: sigma_per_layer length=%d mismatches model layers=%s; will use overlap and fallback to global sigma for others",
                    len(self.sigma_per_layer), str(self._num_layers),
                )
        if self.clip_update_norm_per_layer is not None and self._num_layers is not None:
            if len(self.clip_update_norm_per_layer) != self._num_layers:
                logger.warning(
                    "PUM: clip_update_norm_per_layer length=%d mismatches model layers=%s; using overlap; others un-clipped or use global clip if set",
                    len(self.clip_update_norm_per_layer), str(self._num_layers),
                )

        # Server-side clipping default enablement: on if any sensitivity or explicit C provided
        if server_center_clipping is None:
            self.server_center_clipping = (
                (self.dp_sens_per_layer_l2 is not None and len(self.dp_sens_per_layer_l2) > 0)
                or (self.dp_sens_tot_l2 is not None and self.dp_sens_tot_l2 > 0)
                or (self.center_clip_C_per_layer is not None and len(self.center_clip_C_per_layer) > 0)
                or (self.center_clip_C_global is not None and self.center_clip_C_global > 0)
                or (self.center_clip_ref_model_paths is not None and len(self.center_clip_ref_model_paths) > 0)
            )
        else:
            self.server_center_clipping = _as_bool(server_center_clipping, default=False)

        # Initialize EMA reference and previous published mean (server coords)
        self._theta_ref_sd: Dict[str, torch.Tensor] = _state_dict_tensors(self.model)
        # Preserve pristine base params for public-only quantile computation
        self._theta_base_sd: Dict[str, torch.Tensor] = {k: v.clone().detach().cpu() for k, v in self._theta_ref_sd.items()}
        self._pub_mean_prev_sd: Optional[Dict[str, torch.Tensor]] = None
        # Cache for quantile-based per-layer C_l, computed once if refs provided
        self._center_clip_C_from_quantile: Optional[List[float]] = None

    # ----------------------------
    # DP calibration for sigma
    # ----------------------------
    def _resolve_S_alpha(self, m: int) -> float:
        """Compute S_alpha = sum_k 1/alpha_k^2 used in DP accounting.

        If dp_use_worstcase_alpha is True (default), use the worst-case upper bound m/alpha_min^2.
        Otherwise, approximate the expected value under a uniform distribution on [alpha_min, alpha_max].
        """
        a_min = max(float(self.alpha_min), 1.0)
        a_max = max(float(self.alpha_max), a_min)
        if self.dp_use_worstcase_alpha or a_max <= a_min:
            return m / (a_min ** 2)
        # E[1/alpha^2] for alpha ~ Uniform[a_min, a_max]
        try:
            exp_inv_sq = (1.0 / a_min - 1.0 / a_max) / (a_max - a_min)
            return m * exp_inv_sq
        except ZeroDivisionError:
            return m / (a_min ** 2)

    def _orders_grid(self) -> List[float]:
        if not self.dp_rdp_orders:
            return [1.1, 1.25, 1.5, 2, 3, 4, 5, 8, 10, 16, 32, 64, 128, 256, 512]
        orders = [float(x) for x in self.dp_rdp_orders if float(x) > 1.0]
        return orders or [2.0, 4.0, 8.0, 16.0]

    def _get_per_layer_sens(self, L: int) -> Optional[List[float]]:
        """Resolve per-layer L2 sensitivity bounds Δ̄_{2,ℓ} for ℓ=1..L.

        Priority: explicit dp_sensitivity_per_layer_l2 -> clip_update_norm_per_layer -> uniform split of dp_sensitivity_total_l2.
        """
        if L is None or L <= 0:
            return None
        # 0) If we computed C_l by public-only quantiles, use Δ̄_{2,ℓ} = 2 C_ℓ
        if self._center_clip_C_from_quantile is not None and len(self._center_clip_C_from_quantile) > 0:
            vals = [2.0 * float(c) for c in self._center_clip_C_from_quantile[:L]]
            if len(vals) < L:
                vals = vals + [float(vals[-1])] * (L - len(vals))
            return vals
        # 1) Explicit sensitivities
        if self.dp_sens_per_layer_l2 is not None and len(self.dp_sens_per_layer_l2) > 0:
            vals = [float(x) for x in self.dp_sens_per_layer_l2[:L]]
            if len(vals) < L:
                vals = vals + [float(vals[-1])] * (L - len(vals))
            return vals
        # 2) Use per-layer clipping thresholds as sensitivity proxies
        if self.clip_update_norm_per_layer is not None and len(self.clip_update_norm_per_layer) > 0:
            vals = [
                (float(x) if (x is not None and float(x) > 0) else 0.0)
                for x in self.clip_update_norm_per_layer[:L]
            ]
            if len(vals) < L:
                vals = vals + [0.0] * (L - len(vals))
            return vals
        # 3) Uniform split of total sensitivity across L layers if available
        if self.dp_sens_tot_l2 is not None and self.dp_sens_tot_l2 > 0 and L > 0:
            # Require that sum_l Δ_l^2 = (Δ_tot)^2 with Δ_l all equal => Δ_l = Δ_tot / sqrt(L)
            per = float(self.dp_sens_tot_l2) / math.sqrt(float(L))
            return [per for _ in range(L)]
        return None

    def _calibrate_sigma_from_dp(self) -> None:
        """DP-driven calibration for noise.

        - If per_layer_noise=True, compute self.sigma_per_layer via RDP accounting (equalized or variance-minimizing split).
        - Else compute global self.sigma via single-sigma formula.
        When DP knobs are present, DP calibration takes precedence over manually provided sigma/sigma_per_layer.
        """
        if self.dp_epsilon is None or self.dp_delta is None:
            return

        eps_tgt = float(self.dp_epsilon)
        delta = float(self.dp_delta)
        R = max(int(self.rounds_R), 1)
        m = max(int(self.copies_m), 1)
        S_alpha = self._resolve_S_alpha(m)

        orders = self._orders_grid()

        # Per-layer calibration branch
        if self.per_layer_noise:
            L = int(self._num_layers) if self._num_layers is not None else None
            if not L or L <= 0:
                logger.warning("PUM DP per-layer calibration requested but model layers not found; falling back to single-sigma calibration.")
            else:
                Delta_l = self._get_per_layer_sens(L)
                if Delta_l is None:
                    logger.warning(
                        "PUM DP per-layer calibration needs per-layer or total sensitivity; provide dp_sensitivity_per_layer_l2, clip_update_norm_per_layer, or dp_sensitivity_total_l2. Falling back to single-sigma."
                    )
                else:
                    # Optimize over RDP orders; if allocation='auto', compare equalized vs variance-minimizing and pick smaller variance
                    best_sigmas: Optional[List[float]] = None
                    best_order: Optional[float] = None
                    best_total_var: Optional[float] = None
                    sum_D2 = sum(d * d for d in Delta_l)
                    sum_D = sum(Delta_l)
                    for lam in orders:
                        A = eps_tgt - (math.log(1.0 / delta) / (lam - 1.0))
                        if A <= 0:
                            continue
                        K = (2.0 * A) / (R * lam * max(S_alpha, 1e-12))
                        if K <= 0:
                            continue
                        # equalized: σ_l = κ Δ_l, κ = sqrt(L / K)
                        kappa = math.sqrt(float(L) / K)
                        sig_eq = [kappa * d for d in Delta_l]
                        tot_eq = (kappa * kappa) * sum_D2
                        # variance-minimizing: σ_l = c sqrt(Δ_l), c = sqrt((sum Δ_l) / K)
                        c = math.sqrt(max(sum_D, 0.0) / K)
                        sig_vm = [c * math.sqrt(max(d, 0.0)) for d in Delta_l]
                        tot_vm = (c * c) * sum_D

                        candidates = []
                        if self.dp_per_layer_allocation in ("auto", "equalized"):
                            candidates.append((tot_eq, sig_eq))
                        if self.dp_per_layer_allocation in ("auto", "varmin"):
                            candidates.append((tot_vm, sig_vm))

                        for total_var, sigmas in candidates:
                            if not all(math.isfinite(s) and s >= 0 for s in sigmas):
                                continue
                            if (best_total_var is None) or (total_var < best_total_var):
                                best_total_var = total_var
                                best_sigmas = sigmas
                                best_order = lam

                    if best_sigmas is not None:
                        self.sigma_per_layer = [float(s) for s in best_sigmas]
                        # Ensure we do not add noise to non-layer params in per-layer mode
                        self.sigma = 0.0
                        logger.info(
                            "PUM DP per-layer calibration: eps=%.5g, delta=%g, R=%d, m=%d, S_alpha=%.6g, alloc=%s -> order=%.3g",
                            eps_tgt, delta, R, m, S_alpha, self.dp_per_layer_allocation, best_order or float('nan')
                        )
                        return

        # Fallback or global calibration branch
        # Require total sensitivity for single-sigma calibration
        if self.dp_sens_tot_l2 is None:
            return

        Delta2_tot = float(self.dp_sens_tot_l2)
        best_sigma = None
        best_order = None

        for lam in orders:
            A = eps_tgt - (math.log(1.0 / delta) / (lam - 1.0))
            if A <= 0:
                continue
            sig = Delta2_tot * math.sqrt((R * lam * max(S_alpha, 1e-12)) / (2.0 * A))
            if sig <= 0 or math.isnan(sig) or math.isinf(sig):
                continue
            if (best_sigma is None) or (sig < best_sigma):
                best_sigma = sig
                best_order = lam

        if best_sigma is None:
            logger.warning(
                "PUM DP-calibration failed to find feasible sigma; keep existing sigma(s) (sigma=%s, per-layer=%s)",
                str(self.sigma), str(self.sigma_per_layer is not None),
            )
            return

        # Set global sigma and clear per-layer unless the user explicitly provided per-layer without DP
        self.sigma = float(best_sigma)
        if self.per_layer_noise:
            # If per-layer requested but we fell back to global, clear any prior per-layer overrides
            self.sigma_per_layer = None
        logger.info(
            "PUM DP single-sigma: eps=%.5g, delta=%g, R=%d, m=%d, S_alpha=%.6g, Delta2_tot=%.6g -> sigma=%.6g (order=%.3g)",
            eps_tgt, delta, R, m, S_alpha, Delta2_tot, self.sigma, best_order or float('nan')
        )

    # ----------------------------
    # Name parsing helpers
    # ----------------------------
    @staticmethod
    def _param_layer_index(param_name: str) -> Optional[int]:
        """Extract layer index from a parameter name like 'model.layers.12.attn.q_proj.weight'."""
        # Fast path: expect prefix 'model.layers.'
        prefix = "model.layers."
        if not param_name.startswith(prefix):
            return None
        rest = param_name[len(prefix):]
        # rest begins with '<int>.'
        try:
            dot = rest.find(".")
            if dot <= 0:
                return None
            idx = int(rest[:dot])
            return idx
        except Exception:
            return None

    def _build_per_param_sigma(self) -> Optional[Dict[str, float]]:
        if self.sigma_per_layer is None:
            return None
        # Default for non-layer params: use global sigma if >0, else 0.0
        sigma_other = 0.0 if self.per_layer_noise else (self.sigma if (self.sigma and self.sigma > 0) else 0.0)
        per_param: Dict[str, float] = {}
        for n, _ in _named_params(self.model):
            li = self._param_layer_index(n)
            if li is not None and li < len(self.sigma_per_layer):
                per_param[n] = float(self.sigma_per_layer[li])
            else:
                per_param[n] = float(sigma_other)
        return per_param

    # ----------------------------
    # Server-side DP clipping reference and delta
    # ----------------------------
    def _resolve_center_clip_thresholds(self) -> Tuple[Optional[List[float]], Optional[float]]:
        L = int(self._num_layers) if self._num_layers is not None else None
        C_per_layer: Optional[List[float]] = None
        C_global: Optional[float] = None
        # 0) Prefer quantile-based thresholds if computed
        if self._center_clip_C_from_quantile is not None:
            # Apply round gamma scaling if multi-round
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

    # Public-only quantile computation of C_l
    def _maybe_init_center_C_from_quantile(self) -> None:
        if self._center_clip_C_from_quantile is not None:
            return
        paths = self.center_clip_ref_model_paths
        if not paths or len(paths) == 0:
            return
        if not self._num_layers or self._num_layers <= 0:
            logger.warning("PUM: cannot compute quantile C_l; model layers unknown")
            return

        L = int(self._num_layers)
        # Accumulate per-layer norms across J refs
        norms_per_layer: List[List[float]] = [[] for _ in range(L)]

        # Helper to compute per-layer norms between sd and base
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

        # Try loading via from_pretrained; fallback to pytorch_model.bin
        for pth in paths:
            try:
                # Prefer HF from_pretrained if available on the model class
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
                # Fallback: attempt to load common filename
                candidate_files = [
                    "pytorch_model.bin",
                    "adapter_model.bin",
                    "consolidated.00.pth",
                    "model.safetensors",
                ]
                loaded = False
                import os as _os
                for fn in candidate_files:
                    fp = _os.path.join(pth, fn)
                    if _os.path.exists(fp):
                        try:
                            if fp.endswith(".safetensors"):
                                from safetensors.torch import load_file as _load_sft
                                sd = _load_sft(fp, device="cpu")
                            else:
                                sd = torch.load(fp, map_location="cpu")
                            # If state dict nested
                            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                                sd = sd["state_dict"]
                            # Normalize to parameter keys
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

        # Compute quantiles per layer
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
                if s == 0.0:
                    delta[name] = -q
                elif s == 1.0:
                    delta[name] = torch.zeros_like(q)
                else:
                    delta[name] = (s - 1.0) * q
            elif s_other is not None:
                s = s_other
                if s == 0.0:
                    delta[name] = -q
                elif s == 1.0:
                    delta[name] = torch.zeros_like(q)
                else:
                    delta[name] = (s - 1.0) * q
            else:
                delta[name] = torch.zeros_like(q)

        return delta

    # ----------------------------
    # Step budget alignment
    # ----------------------------
    def _maybe_set_auto_local_max_steps(self) -> None:
        """If requested, set local_max_steps so that local_max_steps * rounds_R ~= N,
        where N is the total training steps of the original single-trainer run.

        N = num_train_epochs * ceil(len(train_dataset) / (bs * accum * world_size))
        """
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
        world = 1
        # Prefer TrainingArguments' derived world size, fallback to env
        try:
            world = int(getattr(self.args, "world_size", None) or int(os.environ.get("WORLD_SIZE", 1)))
        except Exception:
            world = 1

        # Prefer explicit max_steps if provided; else derive from epochs and steps/epoch
        explicit_max_steps = int(getattr(self.args, "max_steps", -1) or -1)
        if explicit_max_steps > 0:
            N = explicit_max_steps
            steps_per_epoch = None
            epochs = None
        else:
            steps_per_epoch = math.ceil(dataset_len / max(bs * accum * world, 1))
            epochs = int(getattr(self.args, "num_train_epochs", 1))
            N = steps_per_epoch * max(epochs, 1)
        # local_max_steps * rounds_R ~= N
        self.local_max_steps = max(1, math.ceil(N / max(self.rounds_R, 1)))
        logger.info(
            f"PUM auto-balance: dataset_len={dataset_len}, bs={bs}, accum={accum}, world={world}, "
            f"steps_per_epoch={steps_per_epoch}, epochs={epochs}, explicit_max_steps={explicit_max_steps} -> N={N}; "
            f"rounds_R={self.rounds_R} -> local_max_steps={self.local_max_steps} (per copy, per round)"
        )

    # ----------------------------
    # Orthogonal reparam (placeholders)
    # ----------------------------
    def _sample_T(self, model: nn.Module, seed: int):
        """
        Sample a per-layer transform T = {layer_idx: {R_sub, perm, inv_perm, dims}}
        - Attention: a shared orthogonal R_sub in head_dim applied per head (q,k,v left-mul; o right-mul)
        - FFN: a permutation P over the intermediate width (gate/up left-mul; down right-mul)

        The transform strictly preserves function for Llama-style architectures.
        """
        if not self.use_orthogonal_reparam:
            return None

        torch.manual_seed(seed)
        T: Dict[int, Dict[str, torch.Tensor]] = {}
        # Try to access Llama-style modules
        try:
            layers = model.model.layers  # type: ignore[attr-defined]
        except Exception:
            # Unknown model layout; disable reparam safely
            return None

        for li, layer in enumerate(layers):
            # Attention dims
            attn = getattr(layer, "self_attn", None)
            mlp = getattr(layer, "mlp", None)
            entry: Dict[str, torch.Tensor] = {}
            if attn is not None:
                # Use a single orthogonal over head_dim, repeated across heads
                head_dim = int(getattr(attn, "head_dim", 0) or (attn.q_proj.weight.shape[0] // getattr(attn, "num_heads", 1)))
                if head_dim > 0:
                    # Random orthogonal via QR (compute in float32 on CPU to avoid bf16 CPU QR issues)
                    A = torch.randn(head_dim, head_dim, device="cpu", dtype=torch.float32)
                    Q, _ = torch.linalg.qr(A)
                    entry["R_sub"] = Q.detach().to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
                    # Store head counts for reshape logic
                    entry["n_q_heads"] = torch.tensor(int(getattr(attn, "num_heads", 0)), device="cpu")
                    entry["n_kv_heads"] = torch.tensor(int(getattr(attn, "num_key_value_heads", getattr(attn, "num_heads", 0))), device="cpu")
                    entry["head_dim"] = torch.tensor(head_dim, device="cpu")

            if mlp is not None:
                # Intermediate size inferred from up_proj weight
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

            # Attention block: apply R_sub per head (left mul on q/k/v; right mul on o)
            if attn is not None and "R_sub" in ent:
                R = ent["R_sub"]
                D = int(ent["head_dim"].item())
                Hq = int(ent["n_q_heads"].item())
                Hkv = int(ent["n_kv_heads"].item())

                def _left_mul_block(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    # weight: (H*D, M) -> reshape (H,D,M), apply R @ (D,M) per-head
                    w = weight.view(H, D, -1)
                    if w.device.type == "cpu":
                        w32 = w.float()
                        R32 = R.float()
                        w32 = torch.einsum("ij,hjm->him", R32, w32)
                        return w32.to(weight.dtype).reshape(H * D, -1)
                    else:
                        w = torch.einsum("ij,hjm->him", R, w)
                        return w.reshape(H * D, -1)

                def _right_mul_block(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    # weight: (M, H*D) -> reshape (M,H,D), apply (M,H,D) @ R^T on last dim
                    w = weight.view(-1, H, D)
                    if w.device.type == "cpu":
                        w32 = w.float()
                        R32 = R.float()
                        w32 = torch.einsum("mhd,dj->mhj", w32, R32.T)
                        return w32.to(weight.dtype).reshape(-1, H * D)
                    else:
                        w = torch.einsum("mhd,dj->mhj", w, R.T)
                        return w.reshape(-1, H * D)

                with torch.no_grad():
                    attn.q_proj.weight.copy_(
                        _left_mul_block(attn.q_proj.weight, Hq, D, R)
                    )
                    attn.k_proj.weight.copy_(
                        _left_mul_block(attn.k_proj.weight, Hkv, D, R)
                    )
                    attn.v_proj.weight.copy_(
                        _left_mul_block(attn.v_proj.weight, Hkv, D, R)
                    )
                    attn.o_proj.weight.copy_(
                        _right_mul_block(attn.o_proj.weight, Hq, D, R)
                    )

            # FFN block: permutation over intermediate width
            if mlp is not None and "perm" in ent and "inv_perm" in ent:
                P = ent["perm"].long()
                with torch.no_grad():
                    # gate/up: left-multiply by P -> row reindex
                    mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[P, :])
                    mlp.up_proj.weight.copy_(mlp.up_proj.weight[P, :])
                    # LLaMA MLPs typically are bias=False, but handle if exists
                    if getattr(mlp.gate_proj, "bias", None) is not None:
                        mlp.gate_proj.bias.copy_(mlp.gate_proj.bias[P])
                    if getattr(mlp.up_proj, "bias", None) is not None:
                        mlp.up_proj.bias.copy_(mlp.up_proj.bias[P])
                    # down: right-multiply by P^T -> column reindex by inv_perm
                    invP = ent["inv_perm"].long()
                    mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, invP])

    def _apply_T_inv_to_update(self, update: Dict[str, torch.Tensor], T) -> Dict[str, torch.Tensor]:
        if not T:
            return update

        out: Dict[str, torch.Tensor] = dict(update)

        def _maybe(name: str):
            return name in out and isinstance(out[name], torch.Tensor)

        for li, ent in T.items():
            # Build common name prefix for this layer
            base = f"model.layers.{li}."

            # Attention inverse: q/k/v were left-multiplied by R -> delta preimage = R^T @ delta
            # o was right-multiplied by R^T -> delta preimage = delta @ R
            if "R_sub" in ent:
                R = ent["R_sub"]
                D = int(ent["head_dim"].item())
                Hq = int(ent["n_q_heads"].item())
                Hkv = int(ent["n_kv_heads"].item())

                def _left_inv(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    w = weight.view(H, D, -1)
                    if w.device.type == "cpu":
                        w32 = w.float()
                        RT32 = R.float().T
                        w32 = torch.einsum("ij,hjm->him", RT32, w32)
                        return w32.to(weight.dtype).reshape(H * D, -1)
                    else:
                        w = torch.einsum("ij,hjm->him", R.T, w)
                        return w.reshape(H * D, -1)

                def _right_inv(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    w = weight.view(-1, H, D)
                    if w.device.type == "cpu":
                        w32 = w.float()
                        R32 = R.float()
                        w32 = torch.einsum("mhd,dj->mhj", w32, R32)
                        return w32.to(weight.dtype).reshape(-1, H * D)
                    else:
                        w = torch.einsum("mhd,dj->mhj", w, R)
                        return w.reshape(-1, H * D)

                q_w = base + "self_attn.q_proj.weight"
                k_w = base + "self_attn.k_proj.weight"
                v_w = base + "self_attn.v_proj.weight"
                o_w = base + "self_attn.o_proj.weight"
                if _maybe(q_w):
                    out[q_w] = _left_inv(out[q_w], Hq, D, R)
                if _maybe(k_w):
                    out[k_w] = _left_inv(out[k_w], Hkv, D, R)
                if _maybe(v_w):
                    out[v_w] = _left_inv(out[v_w], Hkv, D, R)
                if _maybe(o_w):
                    out[o_w] = _right_inv(out[o_w], Hq, D, R)

            # FFN inverse: gate/up were left-multiplied by P -> delta preimage = P^T rows
            # down was right-multiplied by P^T -> delta preimage = columns by P
            if "perm" in ent and "inv_perm" in ent:
                P = ent["perm"].long()
                invP = ent["inv_perm"].long()
                g_w = base + "mlp.gate_proj.weight"
                u_w = base + "mlp.up_proj.weight"
                d_w = base + "mlp.down_proj.weight"
                g_b = base + "mlp.gate_proj.bias"
                u_b = base + "mlp.up_proj.bias"
                if _maybe(g_w):
                    out[g_w] = out[g_w][invP, :]
                if _maybe(u_w):
                    out[u_w] = out[u_w][invP, :]
                if _maybe(g_b):
                    out[g_b] = out[g_b][invP]
                if _maybe(u_b):
                    out[u_b] = out[u_b][invP]
                if _maybe(d_w):
                    out[d_w] = out[d_w][:, P]

        return out

    # ----------------------------
    # Helpers
    # ----------------------------
    def _make_inner_args(self, output_dir: str) -> TrainingArguments:
        # Start from a deepcopy of outer args to keep devices/precision consistent
        inner_args = copy.deepcopy(self.args)
        # Ensure no checkpointing/logging side-effects
        inner_args.output_dir = output_dir
        inner_args.save_strategy = "no"
        inner_args.eval_strategy = "no"
        inner_args.do_eval = False
        inner_args.logging_dir = os.path.join(output_dir, "logs")
        inner_args.report_to = [] if getattr(inner_args, "report_to", None) is None else inner_args.report_to
        inner_args.num_train_epochs = self.local_epochs
        if self.local_max_steps is not None:
            inner_args.max_steps = self.local_max_steps
        # Use a different seed so dataloader shuffling differs per copy
        inner_args.seed = (self._base_seed + 17) % (2**31 - 1)
        return inner_args

    def _instantiate_inner_trainer(self, model: nn.Module, args: TrainingArguments):
        inner_cls = _resolve_inner_trainer(self.inner_handler)
        # Inner trainer gets same datasets/collator/evaluators/template_args
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

    def _clip_update(self, update: Dict[str, torch.Tensor]):
        # Prefer per-layer clipping if provided
        if self.clip_update_norm_per_layer is not None and self._num_layers:
            L = min(int(self._num_layers), len(self.clip_update_norm_per_layer))
            # First pass: compute per-layer squared norms
            layer_sq = [0.0 for _ in range(L)]
            for name, v in update.items():
                li = self._param_layer_index(name)
                if li is None or li >= L:
                    continue
                layer_sq[li] += v.detach().float().pow(2).sum().item()
            # Compute scales per layer
            scales = [1.0 for _ in range(L)]
            for li in range(L):
                thr = float(self.clip_update_norm_per_layer[li])
                if thr is None or thr <= 0:
                    continue
                norm = math.sqrt(max(layer_sq[li], 1e-12))
                if norm > thr:
                    scales[li] = thr / (norm + 1e-12)
            # Apply scaling per layer
            out: Dict[str, torch.Tensor] = {}
            for name, v in update.items():
                li = self._param_layer_index(name)
                if li is not None and li < L:
                    s = scales[li]
                    if s != 1.0:
                        out[name] = v * s
                    else:
                        out[name] = v
                else:
                    out[name] = v
            # Optionally apply global clip to 'other' params if requested
            if self.clip_update_norm is not None:
                # Compute norm over 'other' params and scale them uniformly
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

        # Fallback: global L2 clipping
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
    def train(self, resume_from_checkpoint: Optional[str] = None, **kwargs):
        self.model.train()

        # Optionally set local_max_steps to match original total steps budget
        self._maybe_set_auto_local_max_steps()

        # If requested and references provided, compute quantile-based C_l once (before DP calibration)
        if self.server_center_clipping:
            self._maybe_init_center_C_from_quantile()

        # Optionally set sigma/sigma_per_layer from DP budget (DP takes precedence over manual if provided)
        self._calibrate_sigma_from_dp()

        for r in range(1, self.rounds_R + 1):
            # Update clipping reference via EMA of last round's published mean (server coords)
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

            # Re-run DP calibration each round to pick up newly available C_l and updated S_alpha
            self._calibrate_sigma_from_dp()

            # Draw zero-sum base noises
            per_param_sigma = self._build_per_param_sigma()
            base_noises = _generate_zero_sum_noises(
                self.model, self.copies_m, self.sigma, per_param_sigma=per_param_sigma
            )
            # Sample secret alphas
            alphas: List[float] = []
            g = torch.Generator(device="cpu")
            g.manual_seed(self._base_seed + 9973 * r)
            for _ in range(self.copies_m):
                if self.alpha_max == self.alpha_min:
                    alphas.append(self.alpha_min)
                else:
                    u = torch.rand(1, generator=g).item()
                    alphas.append(self.alpha_min + u * (self.alpha_max - self.alpha_min))

            # Streaming stats
            S0 = 0.0
            S1 = _zero_like_param_dict(self.model)
            # Per-round publication mean accumulator in server coordinates (aligned T^{-1} of published params)
            pub_sum_server = _zero_like_param_dict(self.model)
            # Per-round per-layer norms from aligned published models for quantile C_l update
            round_norms_per_layer: Optional[List[List[float]]] = (
                [[] for _ in range(int(self._num_layers))]
                if (self._num_layers is not None and int(self._num_layers) > 0)
                else None
            )
            # Server-side center clipping delta (relative to current params)
            cur_sd = _state_dict_tensors(self.model)
            center_delta = None
            if self.server_center_clipping:
                C_per_layer, C_global = self._resolve_center_clip_thresholds()
                center_delta = self._compute_center_clip_delta(cur_sd, self._theta_ref_sd, C_per_layer, C_global)

            for k in range(self.copies_m):
                alpha_k = float(alphas[k])
                # Perturbation for this copy
                eps_k = {n: alpha_k * base_noises[k][n] for (n, _) in _named_params(self.model)}
                # Tiny independent jitter
                xi_k = None
                if self.jitter_tau and self.jitter_tau > 0.0:
                    xi_k = {}
                    for n, p in _named_params(self.model):
                        xi_k[n] = torch.randn_like(p) * float(self.jitter_tau)

                # Create a model copy with perturbation applied
                model_k = copy.deepcopy(self.model)
                # If a functional T is desired, sample and apply (identity by default)
                T_k = (
                    self._sample_T(model_k, seed=self._base_seed + 31 * r + 7 * k)
                    if self.use_orthogonal_reparam
                    else None
                )

                # Apply center shift, epsilon, and jitter in-place on model_k
                if center_delta is not None:
                    _apply_state_dict_delta(model_k, center_delta, scale=1.0)
                _apply_state_dict_delta(model_k, eps_k, scale=1.0)
                if xi_k is not None:
                    _apply_state_dict_delta(model_k, xi_k, scale=1.0)

                # Apply reparameterization (identity now)
                self._apply_T(model_k, T_k)

                # Build inner args and inner trainer (may wrap/modify model)
                inner_out_dir = os.path.join(
                    str(self.args.output_dir), f"pum_round{r}_copy{k+1}"
                )
                inner_args = self._make_inner_args(inner_out_dir)
                inner_trainer = self._instantiate_inner_trainer(model_k, inner_args)

                # Snapshot pre-training params in transformed coordinate (after any trainer wrapping, before train)
                before_sd = _state_dict_tensors(inner_trainer.model)
                # Align published params back to server coordinates via T^{-1} and accumulate for EMA/quantiles
                before_sd_aligned = self._apply_T_inv_to_update(before_sd, T_k)
                with torch.no_grad():
                    for n, _ in _named_params(self.model):
                        pub_sum_server[n].add_(before_sd_aligned[n].to(pub_sum_server[n].device))
                # Collect per-layer norms vs base for quantile-based C_l
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

                # Local unlearning train
                inner_trainer.train()

                # Compute update in transformed coords: Delta^{(k,r)} = theta_k_after - theta_k_before
                after_sd = _state_dict_tensors(inner_trainer.model)
                delta_k = _diff_state_dict(after_sd, before_sd)

                # Invert T (identity for now)
                delta_k = self._apply_T_inv_to_update(delta_k, T_k)

                # Optional clipping
                delta_k = self._clip_update(delta_k)

                # Streaming harmonic-normalized accumulation (intersection of keys)
                w = 1.0 / max(alpha_k, 1e-6)
                S0 += w
                for n, dv in delta_k.items():
                    if n in S1:
                        S1[n].add_(w * dv.to(S1[n].device))

                # Free up memory explicitly
                del inner_trainer
                del model_k
                del after_sd, delta_k
                torch.cuda.empty_cache()

            # Aggregate and apply to global model: theta_r = theta_{r-1} + eta_srv * (S1 / S0)
            inv_S0 = 1.0 / max(S0, 1e-12)
            with torch.no_grad():
                for n, p in _named_params(self.model):
                    update = (S1[n] * inv_S0).to(p.device)
                    p.add_(self.eta_srv * update)

            # Store published mean (server coords) for next-round EMA reference
            m = float(self.copies_m)
            self._pub_mean_prev_sd = {n: (pub_sum_server[n] / m).detach().clone() for n in pub_sum_server}

            # Update quantile-based C_l from this round's aligned published models (public-only)
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

        # End of rounds; switch to eval mode for downstream evaluate()
        self.model.eval()
        return None
