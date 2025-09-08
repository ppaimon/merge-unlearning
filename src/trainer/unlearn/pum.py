import os
import math
import copy
import importlib
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from transformers import TrainingArguments
from trainer.base import FinetuneTrainer


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
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        eta_srv: float = 1.0,
        local_epochs: int = 1,
        local_max_steps: Optional[int] = None,
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
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.eta_srv = float(eta_srv)
        self.local_epochs = int(local_epochs)
        self.local_max_steps = (
            int(local_max_steps) if local_max_steps is not None else None
        )
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
    # Orthogonal reparam (placeholders)
    # ----------------------------
    def _sample_T(self, model: nn.Module, seed: int):
        # Placeholder: identity transform; return opaque handle if needed
        return None

    def _apply_T(self, model: nn.Module, T) -> None:
        # Identity: no-op. Future: implement per-layer orthogonal/permutation transforms.
        return

    def _apply_T_inv_to_update(self, update: Dict[str, torch.Tensor], T) -> Dict[str, torch.Tensor]:
        # Identity for now
        return update

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

                # Build inner args and inner trainer
                inner_out_dir = os.path.join(
                    str(self.args.output_dir), f"pum_round{r}_copy{k+1}"
                )
                inner_args = self._make_inner_args(inner_out_dir)
                inner_trainer = self._instantiate_inner_trainer(model_k, inner_args)

                # Local unlearning train
                inner_trainer.train()

                # Compute update: Delta^{(k,r)} = theta_k_after - (theta_{r-1} + eps_k)
                after_sd = _state_dict_tensors(inner_trainer.model)
                base_plus_eps = {
                    n: p.detach().clone().to(v.device)
                    for (n, v), (_, p) in zip(after_sd.items(), _named_params(self.model))
                }
                for n in base_plus_eps:
                    # base model param
                    base_plus_eps[n] = self.model.get_parameter(n).detach().clone().to(
                        after_sd[n].device
                    )
                    base_plus_eps[n].add_(eps_k[n].to(base_plus_eps[n].device))

                delta_k = _diff_state_dict(after_sd, base_plus_eps)  # aligned with model_k coords

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
                del after_sd, base_plus_eps, delta_k
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

