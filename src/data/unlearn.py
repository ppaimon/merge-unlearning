# src/data/unlearn.py
from typing import Optional, Dict, Any
import torch
from torch.utils.data import Dataset, get_worker_info
import torch.distributed as dist


__all__ = ["ForgetRetainDataset"]


def _safe_get_rank() -> int:
    """Return current distributed rank (0 if not initialized)."""
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


class ForgetRetainDataset(Dataset):
    """
    Wraps the forget/retain datasets into an unlearning dataset with robust per-rank/per-worker
    randomness for the non-anchor side. This avoids identical retain sampling across ranks.

    Args:
        forget (Dataset): Forget dataset
        retain (Dataset): Retain dataset
        anchor (str): "forget" or "retain". The anchored side is indexed by `idx`;
                      the other side is sampled uniformly at random.
        seed (Optional[int]): Optional base seed to initialize the dataset's private RNG.
                              If None, uses torch.initial_seed() and mixes rank/worker/epoch.
    """

    def __init__(
        self,
        forget: Dataset,
        retain: Optional[Dataset],
        anchor: str = "forget",
        seed: Optional[int] = None,
    ):
        self.forget = forget
        self.retain = retain
        if anchor not in ("forget", "retain"):
            raise NotImplementedError(f"{anchor} can be only 'forget' or 'retain'")
        self.anchor = anchor

        # Private RNG state (per-rank, per-worker, per-epoch)
        self._base_seed = int(seed) if seed is not None else None
        self._rng: Optional[torch.Generator] = None
        self._epoch: int = 0  # Can be updated via set_epoch()

    # ----------------------------
    # Optional epoch hook
    # ----------------------------
    def set_epoch(self, epoch: int) -> None:
        """(Optional) Call at the beginning of each epoch to decorrelate sampling across epochs."""
        self._epoch = int(epoch)
        self._rng = None  # force re-init with new epoch mix

    # ----------------------------
    # RNG helpers
    # ----------------------------
    def _ensure_rng(self) -> None:
        """Lazy-init a torch.Generator unique to (rank, worker_id, epoch)."""
        if self._rng is not None:
            return

        base = self._base_seed if self._base_seed is not None else torch.initial_seed()
        rank = _safe_get_rank()
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0

        # Mix rank / worker / epoch into the seed to ensure distinct RNG streams
        mixed = (int(base) + 1000003 * rank + 10007 * worker_id + 9721 * (self._epoch + 1)) % (2**63 - 1)

        g = torch.Generator(device="cpu")
        g.manual_seed(mixed)
        self._rng = g

    def _randint(self, low: int, high: int) -> int:
        self._ensure_rng()
        return torch.randint(low, high, (1,), generator=self._rng).item()

    # ----------------------------
    # Dataset protocol
    # ----------------------------
    def __len__(self) -> int:
        if self.anchor == "forget":
            if self.forget is None:
                raise ValueError("forget dataset can't be None when anchor='forget'")
            return len(self.forget)
        elif self.anchor == "retain":
            if self.retain is None:
                raise ValueError("retain dataset can't be None when anchor='retain'")
            return len(self.retain)
        else:
            # Guarded at __init__, but keep for safety
            raise NotImplementedError(f"{self.anchor} can be only 'forget' or 'retain'")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {}

        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            if self.retain is not None and len(self.retain) > 0:
                retain_idx = self._randint(0, len(self.retain))
                item["retain"] = self.retain[retain_idx]

        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            if self.forget is not None and len(self.forget) > 0:
                forget_idx = self._randint(0, len(self.forget))
                item["forget"] = self.forget[forget_idx]

        return item
