# File: src/trainer/unlearn/reparam_epochwise.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn

def _block_diag(mats: Sequence[torch.Tensor]) -> torch.Tensor:
    # torch.block_diag exists but we want graceful empty handling
    if len(mats) == 0:
        raise ValueError("No blocks for block_diag.")
    return torch.block_diag(*mats)

def _rand_rope_commuting_R(dh: int, g: torch.Generator, device, dtype) -> torch.Tensor:
    """
    Build an orthogonal matrix R \in O(dh) that *commutes with 2D RoPE pairs*
    by composing independent 2x2 rotations over pairs (0,1), (2,3), ...
    """
    if dh % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE commuting rotation, got {dh}")
    n_pairs = dh // 2
    blocks = []
    # Random angles in [0, 2pi)
    angles = torch.rand(n_pairs, generator=g, device=device, dtype=torch.float32) * (2 * math.pi)
    for theta in angles:
        c = torch.cos(theta).item()
        s = torch.sin(theta).item()
        blocks.append(torch.tensor([[c, -s], [s, c]], device=device, dtype=torch.float32))
    R = _block_diag(blocks).to(dtype=dtype)
    # Optional extra sign flip on pairs keeps orthogonality and RoPE commutation
    signs = (torch.randint(0, 2, (n_pairs,), generator=g, device=device) * 2 - 1).to(dtype=dtype)
    Sblocks = []
    for s in signs:
        Sblocks.append(torch.diag(torch.stack([s, s]).to(dtype=dtype)))
    S = _block_diag(Sblocks)
    return (R @ S).to(dtype=dtype)

def _as_permutation(n: int, g: torch.Generator, device) -> torch.Tensor:
    """Random permutation indices 0..n-1"""
    return torch.randperm(n, generator=g, device=device)

@dataclass
class _AttnHandles:
    q: nn.Linear
    k: nn.Linear
    v: nn.Linear
    o: nn.Linear
    num_heads: int
    head_dim: int
    hidden_size: int

@dataclass
class _MlpHandles:
    gate: nn.Linear   # gate_proj
    up: nn.Linear     # up_proj
    down: nn.Linear   # down_proj
    inter_dim: int

class ReparamManager:
    """
    Epoch-wise, linear, data-independent reparameterizations + exact inverse.
    Implements the transforms stated in your PUM Methodology (Sec. Reparameterizations).
    """
    def __init__(self, model: nn.Module, ops: Tuple[str, ...] = ("attn", "mlp")):
        self.model = model
        self.ops = set(ops)
        self._attn_R: List[torch.Tensor] = []
        self._mlp_perm: List[torch.Tensor] = []
        self._active = False
        self._attn_layers: List[_AttnHandles] = list(self._iter_attn_layers())
        self._mlp_layers:  List[_MlpHandles]  = list(self._iter_mlp_layers())

    # --------- discovery of layers for HF LLaMA/Mistral-style blocks ----------
    def _iter_decoder_layers(self) -> Iterable[nn.Module]:
        # LLaMA: model.model.layers
        root = getattr(self.model, "model", self.model)
        layers = getattr(root, "layers", None)
        if isinstance(layers, (list, tuple)):
            for x in layers:
                yield x
            return
        # GPT-NeoX/MPT style fallbacks
        tr = getattr(self.model, "transformer", None)
        hs = getattr(tr, "h", None)
        if isinstance(hs, (list, tuple)):
            for x in hs:
                yield x
            return
        # Last resort: walk modules; choose those that have both attn & mlp
        for m in self.model.modules():
            if hasattr(m, "self_attn") and hasattr(m, "mlp"):
                yield m

    def _iter_attn_layers(self) -> Iterable[_AttnHandles]:
        for layer in self._iter_decoder_layers():
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None) or getattr(layer, "attention", None)
            if attn is None:
                continue
            q = getattr(attn, "q_proj", None)
            k = getattr(attn, "k_proj", None)
            v = getattr(attn, "v_proj", None)
            o = getattr(attn, "o_proj", None)
            if any(x is None for x in (q, k, v, o)):
                continue
            num_heads = getattr(attn, "num_heads", None) or getattr(attn, "n_heads", None)
            head_dim = getattr(attn, "head_dim", None)
            hidden_size = getattr(attn, "hidden_size", None) or (num_heads * head_dim if (num_heads and head_dim) else None)
            if not (num_heads and head_dim and hidden_size):
                continue
            yield _AttnHandles(q, k, v, o, int(num_heads), int(head_dim), int(hidden_size))

    def _iter_mlp_layers(self) -> Iterable[_MlpHandles]:
        for layer in self._iter_decoder_layers():
            mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
            if mlp is None:
                continue
            gate = getattr(mlp, "gate_proj", None) or getattr(mlp, "w1", None)
            up   = getattr(mlp, "up_proj", None)   or getattr(mlp, "w3", None)
            down = getattr(mlp, "down_proj", None) or getattr(mlp, "w2", None)
            if any(x is None for x in (gate, up, down)):
                continue
            inter_dim = getattr(gate, "out_features", None)
            if not inter_dim:
                continue
            yield _MlpHandles(gate, up, down, int(inter_dim))

    # ------------------------------- API --------------------------------------
    @torch.no_grad()
    def apply(self, seed: int) -> None:
        """
        Apply a fresh reparameterization (ops subset of {"attn","mlp"}).
        Call once at the *start of each epoch*.
        """
        if self._active:
            raise RuntimeError("ReparamManager.apply(): already active. Call invert() before re-applying.")
        self._attn_R.clear()
        self._mlp_perm.clear()
        device = next(self.model.parameters()).device
        dtype  = next(self.model.parameters()).dtype
        g = torch.Generator(device="cpu")  # CPU RNG for determinism across devices
        g.manual_seed(int(seed))

        if "attn" in self.ops:
            for h in self._attn_layers:
                # Build block-diagonal R across heads, each head uses a RoPE-commuting 2x2-rotations block
                R_heads = []
                for _ in range(h.num_heads):
                    R_heads.append(_rand_rope_commuting_R(h.head_dim, g, device, torch.float32))
                Rblk = _block_diag(R_heads).to(device=device, dtype=dtype)  # [hidden_size, hidden_size]
                self._apply_attn_left_right(h, Rblk)
                self._attn_R.append(Rblk)

        if "mlp" in self.ops:
            for h in self._mlp_layers:
                perm = _as_permutation(h.inter_dim, g, device)
                self._apply_mlp_perm(h, perm)
                self._mlp_perm.append(perm)

        self._active = True

    @torch.no_grad()
    def invert(self) -> None:
        """
        Apply the exact inverse of the last apply(). Call once at the *end of the epoch*.
        """
        if not self._active:
            return
        if "mlp" in self.ops:
            for h, perm in zip(self._mlp_layers, self._mlp_perm):
                inv = torch.argsort(perm)
                # inverse permutation
                self._apply_mlp_perm(h, inv)
        if "attn" in self.ops:
            for h, Rblk in zip(self._attn_layers, self._attn_R):
                # Inverse: left-multiply by R^T on q,k,v; right-multiply o by R
                self._apply_attn_left_right(h, Rblk.T, inverse=True)
        self._active = False
        self._attn_R.clear()
        self._mlp_perm.clear()

    # ------------------------------ kernels -----------------------------------
    @staticmethod
    def _apply_attn_left_right(h: _AttnHandles, Rblk: torch.Tensor, inverse: bool = False):
        # Shapes: h.q.weight: [hidden_size, hidden_size] (no bias in LLaMA)
        # Left-multiply q,k,v; right-multiply o
        # Forward equivalence: y = W_O R^T (R W_V) ... (R W_Q) x  == W_O W_V .. W_Q x
        for lin in (h.q, h.k, h.v):
            w = lin.weight.data
            # w' = R w  (left multiply)
            lin.weight.data = Rblk @ w
            if lin.bias is not None:  # usually None in LLaMA
                lin.bias.data = (Rblk @ lin.bias.data)
        # o: w' = w R^T  (right multiply)
        w = h.o.weight.data
        h.o.weight.data = w @ Rblk.T
        if h.o.bias is not None:
            # No change; bias sits on output channels; R acts on input channels of W_O
            pass

    @staticmethod
    def _apply_mlp_perm(h: _MlpHandles, perm: torch.Tensor):
        # gate, up: permute OUT channels (rows)
        h.gate.weight.data = h.gate.weight.data.index_select(0, perm)
        h.up.weight.data   = h.up.weight.data.index_select(0, perm)
        if h.gate.bias is not None:
            h.gate.bias.data = h.gate.bias.data.index_select(0, perm)
        if h.up.bias is not None:
            h.up.bias.data   = h.up.bias.data.index_select(0, perm)
        # down: permute IN channels (columns)
        h.down.weight.data = h.down.weight.data.index_select(1, perm)
        # (down.bias unchanged)










# === Epoch-wise callback for HF Trainer (append to this file) =================
from typing import Optional, Tuple
from transformers import TrainerCallback

class EpochwiseReparamCallback(TrainerCallback):
    """
    HF Trainer callback that:
      - applies a fresh reparameterization at the **start** of each epoch
      - inverts it (projects back) at the **end** of the epoch

    It is data-independent and keeps the vanilla training loop unchanged.
    You must provide a ReparamManager(model, ops=...) in this file.
    """
    def __init__(
        self,
        ops: Tuple[str, ...] = ("attn", "mlp"),
        seed_offset: int = 0,
        reset_optimizer_each_epoch: bool = False,
        set_epoch_on_dataset: bool = True,
    ):
        self.ops = tuple(ops)
        self.seed_offset = int(seed_offset)
        self.reset_optimizer_each_epoch = bool(reset_optimizer_each_epoch)
        self.set_epoch_on_dataset = bool(set_epoch_on_dataset)
        self._mgr: Optional["ReparamManager"] = None  # type: ignore[name-defined]
        self._epoch_idx: int = -1

    # -- lifecycle ---------------------------------------------------------
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if self._mgr is None and model is not None:
            # ReparamManager must exist in this module already
            self._mgr = ReparamManager(model, ops=self.ops)  # type: ignore[name-defined]

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._mgr is None:
            return
        epoch = int(state.epoch or 0)
        self._epoch_idx = epoch

        # (Optional) let dataset reshuffle per epoch if it supports set_epoch
        trainer = kwargs.get("trainer")
        if self.set_epoch_on_dataset and trainer is not None:
            ds = getattr(trainer, "train_dataset", None)
            if hasattr(ds, "set_epoch"):
                try:
                    ds.set_epoch(epoch)
                except Exception:
                    pass

        seed = int(args.seed) + self.seed_offset + epoch
        self._mgr.apply(seed=seed)

        # (Optional) reset optimizer and scheduler to avoid stale moments
        if self.reset_optimizer_each_epoch and trainer is not None:
            try:
                trainer.optimizer = None
                trainer.create_optimizer()
                trainer.lr_scheduler = None
                num_steps = getattr(state, "max_steps", None) or args.max_steps
                trainer.create_scheduler(num_training_steps=num_steps)
            except Exception:
                # Keep training even if recreation fails
                pass

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._mgr is not None:
            self._mgr.invert()

    def on_train_end(self, args, state, control, **kwargs):
        # Safety net for early-stops
        if self._mgr is not None:
            self._mgr.invert()
            self._mgr = None

