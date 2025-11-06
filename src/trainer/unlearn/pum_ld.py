import os
import math
import copy
import importlib
import logging
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
from transformers.trainer_utils import TrainOutput
from transformers import AutoModelForCausalLM
import torch

from trainer.base import FinetuneTrainer

# NEW: PUM–LD engine (no clipping; α‑scaled, linearly dependent copies)
from trainer.unlearn.pum import PUMTrainer as _PUMEngine, PUMConfig as _PUMCfg

from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safe_load_file

logger = logging.getLogger(__name__)

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

        # 取出 base 模型路径（不要传进 PUMConfig）
        base_model_name_or_path = pum_cfg.pop("base_model_name_or_path", None)

        # 构造 dataclass（里面已有 sigma_ref / noise_generator 等）
        pcfg = _PUMCfg(**pum_cfg)

        # 如需任务向量参考，则加载 base 权重并传给 PUMTrainer
        if pcfg.sigma_ref.lower() == "task_vector":
            if base_model_name_or_path is None:
                raise ValueError(
                    "sigma_ref='task_vector' 需要提供 pum_cfg.base_model_name_or_path"
                )
                        # NEW: 避开 from_pretrained（会被 ZeRO-3 拦截成空分片），
            # 直接把 safetensors/bin 权重加载为 CPU state_dict。
            def _load_base_state_dict_cpu(path_or_repo: str) -> Dict[str, torch.Tensor]:
                # 解析为本地目录；若是 Hub 仓库名则拉取到缓存
                local_dir = path_or_repo
                if not os.path.isdir(local_dir):
                    local_dir = snapshot_download(
                        path_or_repo,
                        allow_patterns=[
                            "model.safetensors",
                            "model-*.safetensors",
                            "pytorch_model.bin",
                            "pytorch_model-*.bin",
                        ],
                    )

                # 优先使用 safetensors 分片
                st_files = sorted([f for f in os.listdir(local_dir) if f.endswith(".safetensors")])
                state: Dict[str, torch.Tensor] = {}
                if st_files:
                    for fn in st_files:
                        shard = safe_load_file(os.path.join(local_dir, fn), device="cpu")
                        state.update(shard)
                else:
                    # 回退到 .bin 分片
                    bin_files = sorted([f for f in os.listdir(local_dir) if f.endswith(".bin")])
                    assert bin_files, f"No model weights found under {local_dir}"
                    for fn in bin_files:
                        shard = torch.load(os.path.join(local_dir, fn), map_location="cpu")
                        state.update(shard)

                # 统一保证 CPU 与独立存储
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.detach().to("cpu")
                return state

            base_sd = _load_base_state_dict_cpu(base_model_name_or_path)
            self._pum_engine = _PUMEngine(self.model, pcfg, base_ref_sd=base_sd)

        else:
            self._pum_engine = _PUMEngine(self.model, pcfg)



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
            batch = self._prepare_inputs(batch)['forget']
            outputs = client_model(input_ids=batch.get('input_ids'), attention_mask=batch.get('attention_mask'), labels=batch.get('labels'))

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Fallback CE if the model did not compute loss internally
                logits = outputs.logits
                labels = batch.get("labels")
                if labels is None:
                    raise RuntimeError("Batch has no 'labels' for loss computation.")
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            if not loss.requires_grad:
                raise RuntimeError(f"Loss does not require grad. Cannot backward.")
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        client_model.to('cpu')
        # Return Δ = after - base
        after = client_model.state_dict()
        delta = {k: (after[k] - theta_pub_state_dict[k].cpu()) for k in theta_pub_state_dict.keys()}
        return delta

    # --------------------------
    # One PUM–LD round + save
    # --------------------------

    def train(self, *args, **kwargs) -> TrainOutput:
        # Run the engine (updates self.model in-place)
        # ... in __init__ or train(), define rounds (default 1)
        R = getattr(self, "rounds", 1)
        # if you store rounds under method_args, do:
        # R = getattr(self.method_args, "rounds", 1)

        for r in range(R):
            new_state, bar_delta = self._pum_engine.run_round(self._client_unlearn_fn, round_idx=r)
            # logging/saving per round if you like


        # -------------------------
        # Determine main process (rank 0) for saving
        # -------------------------
        def is_main_process(trainer_self) -> bool:
            # Preferred HF helper if present
            try:
                return trainer_self.is_world_process_zero()
            except Exception:
                pass
            # Fallback to torch.distributed if initialized
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    return torch.distributed.get_rank() == 0
                except Exception:
                    return True
            return True

        main_proc = is_main_process(self)

        # -------------------------
        # Resolve explicit output_dir (absolute)
        # -------------------------
        output_dir = getattr(self.args, "output_dir", None)
        if output_dir is None:
            # fallback to something explicit inside cwd
            output_dir = os.path.join(os.getcwd(), "pum_ld_output")
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        # Diagnostic logging (very helpful).
        try:
            rank = torch.distributed.get_rank() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 0
        except Exception:
            rank = 0
        logger.info(f"[PUM_LD] Saving: main_proc={main_proc}, rank={rank}, cwd={os.getcwd()}, resolved_output_dir={output_dir}")
        logger.info(f"[PUM_LD] self.args.output_dir (raw) = {getattr(self.args, 'output_dir', None)}")

        # -------------------------
        # Save only on the main process
        # -------------------------
        if main_proc:
            # explicit model.save_pretrained (writes pytorch_model.bin + config.json)
            try:
                self.model.save_pretrained(output_dir)
            except Exception as e:
                # some models don't implement save_pretrained: fallback to state_dict
                logger.warning("[PUM_LD] model.save_pretrained failed: %s. Falling back to state_dict save.", str(e))
                torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

            # ensure config is saved
            try:
                if hasattr(self.model, "config"):
                    self.model.config.save_pretrained(output_dir)
            except Exception as e:
                logger.warning("[PUM_LD] model.config.save_pretrained failed: %s", str(e))

            # tokenizer if present
            try:
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
            except Exception as e:
                logger.warning("[PUM_LD] tokenizer.save_pretrained failed: %s", str(e))

            # write a tiny manifest file to prove we saved here (helpful for debugging)
            try:
                with open(os.path.join(output_dir, "PUM_LD_SAVE_MANIFEST.txt"), "w") as f:
                    f.write(f"saved_by: PUM_LD\nrank: {rank}\nmain_proc: {main_proc}\n")
            except Exception:
                pass
        else:
            logger.info("[PUM_LD] Not main process — skipping actual disk save.")

        # Minimal training metadata so your scripts/postcheck remain happy
        self.state.global_step = max(self.state.global_step, 1)
        metrics = {"loss": 0.0, "pum_ld_rounds": 1}
        self.log(metrics)
        try:
            self.save_state()
        except Exception:
            pass

        return TrainOutput(global_step=self.state.global_step, training_loss=0.0, metrics=metrics)