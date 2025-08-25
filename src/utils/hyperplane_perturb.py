# src/utils/hyperplane_perturb.py
import torch
from contextlib import contextmanager
from typing import Dict, List

class HyperplanePerturb:
    """
    生成 n 个扰动 {r_i}，满足 sum_i alpha_i r_i = 0（默认 n=3、alpha 均匀），
    并提供 apply(i) 上下文：进入时把 r_i 原地加到模型参数，退出时减回去。
    """
    def __init__(
        self,
        model: torch.nn.Module,
        n_points: int = 3,
        alpha: List[float] = None,
        radius: float = 1e-2,
        seed: int = 1234,
        per_tensor_radius: bool = True,
    ):
        assert n_points >= 2
        self.model = model
        self.n = int(n_points)
        if alpha is None:
            self.alpha = torch.full((self.n,), 1.0 / self.n, dtype=torch.float32)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
            assert abs(self.alpha.sum().item() - 1.0) < 1e-6, "sum(alpha) must be 1."
            assert len(self.alpha) == self.n
        self.radius = float(radius)
        self.seed = int(seed)
        self.per_tensor_radius = bool(per_tensor_radius)
        self._make_perturbations()

    @torch.no_grad()
    def _make_perturbations(self):
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        params = [p for p in self.model.parameters() if p.requires_grad]
        # 采样原始噪声 r'_i
        raw: List[Dict[torch.nn.Parameter, torch.Tensor]] = [dict() for _ in range(self.n)]
        for p in params:
            shp = p.shape
            for i in range(self.n):
                raw[i][p] = torch.randn(shp, generator=g, dtype=torch.float32, device="cpu")
        # 加权均值投影：r_i = r'_i - sum_j alpha_j r'_j  ⇒  sum alpha_i r_i = 0
        mean = {p: sum(self.alpha[j] * raw[j][p] for j in range(self.n)) for p in params}
        self.r = [dict() for _ in range(self.n)]
        for i in range(self.n):
            for p in params:
                t = raw[i][p] - mean[p]
                self.r[i][p] = t
        # 半径归一：逐 tensor L2 归一到 radius（简单稳妥）
        if self.per_tensor_radius:
            for i in range(self.n):
                for p in params:
                    t = self.r[i][p]
                    nrm = t.norm().clamp_min(1e-12)
                    self.r[i][p] = (t / nrm) * self.radius
        else:
            for i in range(self.n):
                tot = sum(float(self.r[i][p].norm()**2) for p in params) ** 0.5
                s = self.radius / (tot + 1e-12)
                for p in params:
                    self.r[i][p] = self.r[i][p] * s

    @contextmanager
    def apply(self, i: int):
        """进入时把 r_i 加到参数上，退出时减回去。"""
        assert 0 <= i < self.n
        bufs = []
        with torch.no_grad():
            for p, ri_cpu in self.r[i].items():
                if not p.requires_grad:
                    continue
                ri = ri_cpu.to(device=p.device, dtype=p.dtype)
                p.add_(ri)
                bufs.append((p, ri))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, ri in bufs:
                    p.sub_(ri)
