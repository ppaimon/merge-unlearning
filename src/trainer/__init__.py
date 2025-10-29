# src/trainer/__init__.py

import copy
import logging
from typing import Dict, Any, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import TrainOutput

from trainer.base import FinetuneTrainer
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.undial import UNDIAL
from trainer.unlearn.ceu import CEU
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.wga import WGA
from trainer.unlearn.pdu import PDU

# NEW: PUM–LD engine (no clipping; α‑scaled, linearly dependent copies)
from trainer.unlearn.pum import PUMTrainer as _PUMEngine, PUMConfig as _PUMCfg

# NEW: epoch-wise reparameterization callback
from trainer.unlearn.reparam_epochwise import EpochwiseReparamCallback

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        dataset_len = len(dataset)
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len) // (batch_size * grad_accum_steps * num_devices)
        )
    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    """
    Loads the trainer from registry + wires args/method_args from Hydra config.

    IMPORTANT: For PUM–LD we allow configs to carry `trainer.pum_cfg.*`.
    To avoid changing your YAML structure, we forward `trainer.pum_cfg`
    into `method_args.pum_cfg` here so the PUM_LD class receives it.
    """
    trainer_args = trainer_cfg.args
    # Base method_args (per your repo design)
    method_args = trainer_cfg.get("method_args", {})
    # --- NEW: also forward top-level `pum_cfg` (if present) into method_args
    pum_cfg = trainer_cfg.get("pum_cfg", None)
    if pum_cfg is not None:
        # Do not overwrite if user also provided method_args.pum_cfg explicitly
        if "pum_cfg" not in method_args:
            method_args = dict(method_args)
            method_args["pum_cfg"] = pum_cfg
            logger.info("Forwarded trainer.pum_cfg into method_args.pum_cfg for handler wiring.")

    trainer_args = load_trainer_args(trainer_args, train_dataset)

    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(f"{trainer_handler_name} handler not set")
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(f"{trainer_handler_name} not implemented or not registered")

    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    logger.info(f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}")

    # -------- NEW: optionally register epoch-wise reparam callback ----------
    reparam_cfg = trainer_cfg.get("reparam", None)
    if reparam_cfg is not None and reparam_cfg.get("enable", False):
        cb = EpochwiseReparamCallback(
            ops=tuple(reparam_cfg.get("ops", ["attn", "mlp"])),
            seed_offset=int(reparam_cfg.get("seed_offset", 0)),
            reset_optimizer_each_epoch=bool(reparam_cfg.get("reset_optimizer_each_epoch", False)),
            set_epoch_on_dataset=bool(reparam_cfg.get("set_epoch_on_dataset", True)),
        )
        try:
            trainer.add_callback(cb)
            logger.info("EpochwiseReparamCallback registered.")
        except Exception as e:
            logger.warning(f"Failed to register EpochwiseReparamCallback: {e}")
    # -----------------------------------------------------------------------

    return trainer, trainer_args


# ==========================================================================================
# NEW: Registry-friendly wrapper that uses the PUM–LD engine inside a FinetuneTrainer shell
# ==========================================================================================
class PUM_LD(FinetuneTrainer):
    """
    Registry handler for your PUM–LD method.

    - Accepts `pum_cfg` (forwarded from trainer.pum_cfg by load_trainer).
    - Uses your engine `_PUMEngine` with `_PUMCfg`.
    - Runs a single PUM–LD round, saves the resulting model for downstream eval.
    - Produces a minimal TrainOutput to keep your pipeline consistent.
    """

    def __init__(
        self,
        *args,
        pum_cfg: Optional[DictConfig] | Optional[dict] = None,
        # local client unlearning hyperparams (can override in trainer.method_args)
        local_steps: int = 10,
        local_lr: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Convert pum_cfg (DictConfig -> dict) then dataclass
        if pum_cfg is None:
            pum_cfg = {}
        if isinstance(pum_cfg, DictConfig):
            pum_cfg = OmegaConf.to_container(pum_cfg, resolve=True)
        assert isinstance(pum_cfg, dict), "pum_cfg must be a DictConfig or dict"

        self._pum_engine = _PUMEngine(self.model, _PUMCfg(**pum_cfg))
        self._pum_local_steps = int(local_steps)
        self._pum_local_lr = float(local_lr)

        logger.info(
            "PUM_LD initialized: local_steps=%d, local_lr=%.2e; pum_cfg=%s",
            self._pum_local_steps, self._pum_local_lr, str(pum_cfg)
        )

    # --------------------------
    # Client-side unlearning fn
    # --------------------------
    def _client_unlearn_fn(self, theta_pub_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Runs a few local steps on the forget (train) dataset starting from theta_pub.
        Returns Δ relative to theta_pub (same param space).
        """
        # Build a detached copy with published weights
        client_model = copy.deepcopy(self.model)
        client_model.load_state_dict(theta_pub_state_dict, strict=True)
        device = self.args.device if hasattr(self.args, "device") else next(client_model.parameters()).device
        client_model.to(device)
        client_model.train()

        optimizer = torch.optim.AdamW(client_model.parameters(), lr=self._pum_local_lr)

        dl = self.get_train_dataloader()  # by design, your train_dataset is the forget split
        it = iter(dl)

        steps = self._pum_local_steps
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            # Move to device (use Trainer's utility to stay consistent)
            batch = self._prepare_inputs(batch)

            # Most HF causal LMs compute loss if labels are given
            outputs = client_model(**batch)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Fallback CE if the model did not compute loss internally
                logits = outputs.logits
                labels = batch.get("labels", None)
                if labels is None:
                    raise RuntimeError("Batch has no 'labels' for loss computation.")
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Return Δ = after - base
        after = client_model.state_dict()
        delta = {k: (after[k] - theta_pub_state_dict[k]) for k in theta_pub_state_dict.keys()}
        return delta

    # --------------------------
    # One PUM–LD round + save
    # --------------------------
    def train(self, *args, **kwargs) -> TrainOutput:
        # Run the engine (this updates self.model in-place)
        new_state, bar_delta = self._pum_engine.run_round(self._client_unlearn_fn)

        # Save artifacts so eval can load from run_dir
        # (save_model persists config, tokenizer if set)
        self.save_model()
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.args.output_dir)

        # Minimal training metadata so your scripts/postcheck remain happy
        self.state.global_step = max(self.state.global_step, 1)
        metrics = {"loss": 0.0, "pum_ld_rounds": 1}
        self.log(metrics)
        try:
            self.save_state()
        except Exception:
            pass

        return TrainOutput(global_step=self.state.global_step, training_loss=0.0, metrics=metrics)


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainers
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)

# NEW: Register PUM–LD handler
_register_trainer(PUM_LD)
