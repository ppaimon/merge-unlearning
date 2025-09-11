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
    # return (a - b)
    out = {}
    for k, va in a.items():
        vb = b[k]
        out[k] = va - vb
    return out


def _generate_zero_sum_noises(
    model: nn.Module, m: int, sigma: float, device: Optional[torch.device] = None
) -> List[Dict[str, torch.Tensor]]:
    """Generate m zero-sum, equal-variance Gaussian noises per-parameter.

    Returns a list of length m where each element is a param-name->tensor map.
    If m == 1 or sigma == 0, returns zeros.
    """
    if m <= 1 or sigma <= 0:
        return [_zero_like_param_dict(model, device=device) for _ in range(m)]

    # For each parameter, construct z_k, subtract mean, scale by sqrt(m/(m-1))
    param_names = [n for (n, _) in _named_params(model)]
    noises = [{n: None for n in param_names} for _ in range(m)]

    for n, p in _named_params(model):
        zs = [torch.randn_like(p, device=device or p.device) * sigma for _ in range(m)]
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
        # DP-based noise calibration (single-sigma across layers)
        dp_epsilon: Optional[float] = None,
        dp_delta: Optional[float] = None,
        dp_sensitivity_total_l2: Optional[float] = None,
        dp_rdp_orders: Optional[List[float]] = None,
        dp_use_worstcase_alpha: bool = True,
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        eta_srv: float = 1.0,
        local_epochs: int = 1,
        local_max_steps: Optional[int] = None,
        auto_balance_local_max_steps: bool = False,
        clip_update_norm: Optional[float] = None,
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
        # DP args
        self.dp_epsilon = float(dp_epsilon) if dp_epsilon is not None else None
        self.dp_delta = float(dp_delta) if dp_delta is not None else None
        self.dp_sens_tot_l2 = (
            float(dp_sensitivity_total_l2) if dp_sensitivity_total_l2 is not None else None
        )
        self.dp_rdp_orders = (
            list(dp_rdp_orders) if dp_rdp_orders is not None else None
        )
        self.dp_use_worstcase_alpha = bool(dp_use_worstcase_alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.eta_srv = float(eta_srv)
        self.local_epochs = int(local_epochs)
        self.local_max_steps = (
            int(local_max_steps) if local_max_steps is not None else None
        )
        self.auto_balance_local_max_steps = bool(auto_balance_local_max_steps)
        self.clip_update_norm = clip_update_norm
        self.use_orthogonal_reparam = use_orthogonal_reparam

        if self.alpha_min < 1.0:
            self.alpha_min = 1.0
        if self.alpha_max < self.alpha_min:
            self.alpha_max = self.alpha_min

        if self.copies_m < 1:
            raise ValueError("copies_m must be >= 1")

        # Seed per-round/copy deterministically for reproducibility across ranks
        self._base_seed = int(self.args.seed or 0)

    # ----------------------------
    # DP calibration for sigma
    # ----------------------------
    def _calibrate_sigma_from_dp(self) -> None:
        """Optionally set self.sigma using DP budget (epsilon, delta).

        Uses single-sigma calibration based on RDP composition across layers/rounds
        with a conservative aggregate over secret scalings: S_alpha <= m / alpha_min^2.
        Requires dp_epsilon, dp_delta, and dp_sensitivity_total_l2.
        """
        if self.dp_epsilon is None or self.dp_delta is None or self.dp_sens_tot_l2 is None:
            return

        # If sigma already set (>0), do not override
        if self.sigma and self.sigma > 0:
            logger.info(
                f"PUM DP-calibration skipped: sigma={self.sigma} already provided."
            )
            return

        eps_tgt = float(self.dp_epsilon)
        delta = float(self.dp_delta)
        R = max(int(self.rounds_R), 1)
        m = max(int(self.copies_m), 1)
        # Conservative S_alpha upper bound
        S_alpha = m / (max(self.alpha_min, 1.0) ** 2)

        # Orders grid for RDP optimization
        if not self.dp_rdp_orders:
            orders = [
                1.1, 1.25, 1.5, 2, 3, 4, 5, 8, 10, 16, 32, 64, 128, 256, 512
            ]
        else:
            orders = [float(x) for x in self.dp_rdp_orders if float(x) > 1.0]
            if not orders:
                orders = [2.0, 4.0, 8.0, 16.0]

        Delta2_tot = float(self.dp_sens_tot_l2)
        best_sigma = None
        best_alpha = None

        for lam in orders:
            # Feasibility check
            denom_eps = eps_tgt - (math.log(1.0 / delta) / (lam - 1.0))
            if denom_eps <= 0:
                continue
            # From single-sigma formula: sigma >= Delta2_tot * sqrt((R * lam * S_alpha) / (2 * denom_eps))
            sig = Delta2_tot * math.sqrt((R * lam * S_alpha) / (2.0 * denom_eps))
            if sig <= 0 or math.isnan(sig) or math.isinf(sig):
                continue
            if (best_sigma is None) or (sig < best_sigma):
                best_sigma = sig
                best_alpha = lam

        if best_sigma is None:
            logger.warning(
                "PUM DP-calibration failed to find feasible sigma; keep existing sigma=%s",
                self.sigma,
            )
            return

        self.sigma = float(best_sigma)
        logger.info(
            f"PUM DP-calibration: epsilon={eps_tgt}, delta={delta}, R={R}, m={m}, "
            f"alpha_min={self.alpha_min} -> S_alpha<={S_alpha:.4f}, Delta2_tot={Delta2_tot} -> "
            f"sigma={self.sigma:.6g} (RDP order={best_alpha})"
        )

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
                    # Random orthogonal via QR
                    A = torch.randn(head_dim, head_dim, device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
                    # Ensure well-conditioned
                    Q, _ = torch.linalg.qr(A)
                    entry["R_sub"] = Q.detach().to(attn.q_proj.weight.dtype)
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
                    w = torch.einsum("ij,hjm->him", R, w)
                    return w.reshape(H * D, -1)

                def _right_mul_block(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    # weight: (M, H*D) -> reshape (M,H,D), apply (M,H,D) @ R^T on last dim
                    w = weight.view(-1, H, D)
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
                    w = torch.einsum("ij,hjm->him", R.T, w)
                    return w.reshape(H * D, -1)

                def _right_inv(weight: torch.Tensor, H: int, D: int, R: torch.Tensor):
                    w = weight.view(-1, H, D)
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
        if self.clip_update_norm is None:
            return update
        # Compute global L2 norm, then scale if above threshold
        total = 0.0
        for v in update.values():
            total += v.detach().float().pow(2).sum().item()
        norm = math.sqrt(max(total, 1e-12))
        if norm <= self.clip_update_norm:
            return update
        scale = self.clip_update_norm / (norm + 1e-12)
        return {k: v * scale for k, v in update.items()}

    # ----------------------------
    # Main train loop
    # ----------------------------
    def train(self, resume_from_checkpoint: Optional[str] = None, **kwargs):
        self.model.train()

        # Optionally set local_max_steps to match original total steps budget
        self._maybe_set_auto_local_max_steps()

        # Optionally set sigma from DP budget
        self._calibrate_sigma_from_dp()

        for r in range(1, self.rounds_R + 1):
            # Draw zero-sum base noises
            base_noises = _generate_zero_sum_noises(self.model, self.copies_m, self.sigma)
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

            for k in range(self.copies_m):
                alpha_k = float(alphas[k])
                # Perturbation for this copy
                eps_k = {n: alpha_k * base_noises[k][n] for (n, _) in _named_params(self.model)}

                # Create a model copy with perturbation applied
                model_k = copy.deepcopy(self.model)
                # If a functional T is desired, sample and apply (identity by default)
                T_k = self._sample_T(model_k, seed=self._base_seed + 31 * r + 7 * k)

                # Apply epsilon in-place on model_k
                _apply_state_dict_delta(model_k, eps_k, scale=1.0)

                # Apply reparameterization (identity now)
                self._apply_T(model_k, T_k)

                # Snapshot pre-training params in transformed coordinate
                before_sd = _state_dict_tensors(model_k)

                # Build inner args and inner trainer
                inner_out_dir = os.path.join(
                    str(self.args.output_dir), f"pum_round{r}_copy{k+1}"
                )
                inner_args = self._make_inner_args(inner_out_dir)
                inner_trainer = self._instantiate_inner_trainer(model_k, inner_args)

                # Local unlearning train
                inner_trainer.train()

                # Compute update in transformed coords: Delta^{(k,r)} = theta_k_after - theta_k_before
                after_sd = _state_dict_tensors(inner_trainer.model)
                delta_k = _diff_state_dict(after_sd, before_sd)

                # Invert T (identity for now)
                delta_k = self._apply_T_inv_to_update(delta_k, T_k)

                # Optional clipping
                delta_k = self._clip_update(delta_k)

                # Streaming harmonic-normalized accumulation
                w = 1.0 / max(alpha_k, 1e-6)
                S0 += w
                for n in S1:
                    S1[n].add_(w * delta_k[n].to(S1[n].device))

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

        # End of rounds; switch to eval mode for downstream evaluate()
        self.model.eval()
        return None
