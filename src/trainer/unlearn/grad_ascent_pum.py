# src/trainer/unlearn/grad_ascent_pum.py
import torch
from utils.hyperplane_perturb import HyperplanePerturb   # ⚠️按工程补齐：若你的包前缀不是 src/，保持与项目一致
from trainer.unlearn.base import UnlearnTrainer          # 已在你的工程中存在

class GradAscentPUM(UnlearnTrainer):
    """
    PUM: Perturbate -> Unlearning (gradient ascent on forget set) -> Merge (alpha-weighted)
    - 不复制多份模型；使用 in-place 噪声加/减 + autograd.grad 逐点取梯度再合并。
    - 仅训练阶段生效；评估阶段沿用 UnlearnTrainer 的 eval 流程。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 维持 HF/accelerate 的初始化
        # 超参数（最小可运行默认；如需从 Hydra/args 读，按你们风格补齐）
        self._n_points = 3
        self._alpha = [1.0 / self._n_points] * self._n_points
        self._radius = 1e-2
        self._seed = 1234
        # 生成 3 点扰动（如需每个 epoch 刷新，可在 on_epoch_begin 重建）
        self._perturber = HyperplanePerturb(
            model=self.model,
            n_points=self._n_points,
            alpha=self._alpha,
            radius=self._radius,
            seed=self._seed,
            per_tensor_radius=True,
        )
        # 预缓存需要梯度的参数列表，便于 zip 对齐
        self._trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        训练阶段被 HF Trainer 调用：
        - 我们在这里手动把 .grad 填好（PUM 合成梯度）；
        - 然后返回一个 0 标量，让外层 backward 成为 no-op；
        - 外层依然会在合适时机调用 optimizer.step()。
        """
        forget_raw = inputs["forget"]
        forget_inputs = {k: forget_raw[k] for k in ("input_ids", "attention_mask", "labels") if k in forget_raw}

        # α 权重张量（放到模型设备上）
        alpha = torch.tensor(self._alpha, device=self.model.device, dtype=torch.float32)

        # 先清掉已有 grad（HF 外层一般已 zero_grad，这里稳妥起见强制一次）
        for p in self._trainable_params:
            p.grad = None

        last_outputs = None  # 仅用于兼容 return_outputs=True 的返回

        # 逐扰动点：加噪->前向->取 -loss 的 grad->按 α_i 累加到 .grad->撤噪
        for i in range(self._n_points):
            with self._perturber.apply(i):
                outputs = model(**forget_inputs)
                last_outputs = outputs
                # 保证是标量：兼容返回 per-example/per-token 的情况
                loss_raw = outputs.loss
                if isinstance(loss_raw, (tuple, list)):
                    loss_raw = loss_raw[0]
                if hasattr(loss_raw, "dim") and loss_raw.dim() > 0:
                    loss_raw = loss_raw.mean()
                loss_i = -loss_raw  # 升梯度
                grads_i = torch.autograd.grad(
                    loss_i,
                    self._trainable_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )


            ai = float(alpha[i].item())
            for p, g in zip(self._trainable_params, grads_i):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(ai * g)

        # 返回 0 标量（避免外层 backward 影响我们手动写入的梯度）
        dummy = torch.zeros(
            (),
            device=self.model.device,
            dtype=(last_outputs.loss.dtype if last_outputs is not None else torch.float32),
        ).requires_grad_()
        return (dummy, last_outputs) if return_outputs else dummy
