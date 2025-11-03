# src/trainer/__init__.py

import copy
import logging
from typing import Dict, Any, Optional, Tuple

import torch
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

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
from trainer.unlearn.pum_ld import PUM_LD

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
