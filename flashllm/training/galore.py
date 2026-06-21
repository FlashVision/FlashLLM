"""GaLore — Gradient Low-Rank Projection optimizer.

Memory-efficient training by projecting gradients into a low-rank subspace,
reducing optimizer state memory from O(mn) to O(mr + nr) where r << min(m,n).

Reference: https://arxiv.org/abs/2403.03507
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer

from flashllm.utils.logger import get_logger

logger = get_logger(__name__)


class GaLoreProjector:
    """Manages low-rank gradient projections for a parameter.

    Computes and caches the projection matrix via SVD, updating it
    periodically to track the changing gradient subspace.

    Args:
        rank: Rank of the projection.
        update_interval: Steps between projection matrix updates.
        scale: Scaling factor for the projected gradient.
    """

    def __init__(self, rank: int, update_interval: int = 200, scale: float = 1.0):
        self.rank = rank
        self.update_interval = update_interval
        self.scale = scale
        self._projector: Optional[torch.Tensor] = None
        self._step = 0

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """Project a full gradient into the low-rank subspace.

        Args:
            grad: Full gradient tensor of shape (m, n).

        Returns:
            Projected gradient of shape (rank, n) or (m, rank).
        """
        if grad.dim() != 2:
            return grad

        self._step += 1

        if self._projector is None or self._step % self.update_interval == 0:
            self._update_projector(grad)

        if grad.shape[0] >= grad.shape[1]:
            return self._projector.t() @ grad
        else:
            return grad @ self._projector

    def project_back(self, low_rank_grad: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Project low-rank gradient back to full space.

        Args:
            low_rank_grad: Low-rank gradient.
            original_shape: Shape of the original gradient.

        Returns:
            Full-rank gradient.
        """
        if len(original_shape) != 2 or self._projector is None:
            return low_rank_grad

        if original_shape[0] >= original_shape[1]:
            return self._projector @ low_rank_grad * self.scale
        else:
            return low_rank_grad @ self._projector.t() * self.scale

    def _update_projector(self, grad: torch.Tensor):
        """Update the projection matrix via SVD."""
        with torch.no_grad():
            if grad.shape[0] >= grad.shape[1]:
                U, _, _ = torch.svd_lowrank(grad.float(), q=self.rank)
                self._projector = U[:, :self.rank].to(grad.dtype)
            else:
                _, _, V = torch.svd_lowrank(grad.float(), q=self.rank)
                self._projector = V[:, :self.rank].to(grad.dtype)


class GaLoreAdamW(Optimizer):
    """AdamW optimizer with GaLore gradient projection.

    For parameters in `galore_params`, gradients are projected into a
    low-rank subspace before the Adam update, then projected back.
    Other parameters use standard AdamW.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate.
        betas: Adam beta coefficients.
        eps: Adam epsilon.
        weight_decay: Weight decay coefficient.
        rank: GaLore projection rank.
        update_proj_gap: Steps between projection matrix updates.
        scale: Gradient scaling factor.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            rank=rank, update_proj_gap=update_proj_gap, scale=scale,
        )
        super().__init__(params, defaults)
        self._projectors: Dict[int, GaLoreProjector] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_galore = group.get("use_galore", False)
            rank = group.get("rank", 128)
            update_proj_gap = group.get("update_proj_gap", 200)
            scale = group.get("scale", 1.0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                if use_galore and grad.dim() == 2:
                    param_id = id(p)
                    if param_id not in self._projectors:
                        self._projectors[param_id] = GaLoreProjector(
                            rank=rank, update_interval=update_proj_gap, scale=scale,
                        )
                    projector = self._projectors[param_id]
                    original_shape = grad.shape
                    grad = projector.project(grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])

                update = exp_avg / denom

                if use_galore and grad.dim() == 2:
                    update = projector.project_back(update, original_shape)

                p.data.add_(update, alpha=-step_size)

        return loss


def setup_galore_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    rank: int = 128,
    update_proj_gap: int = 200,
    scale: float = 1.0,
    weight_decay: float = 0.01,
    target_modules: Optional[List[str]] = None,
) -> GaLoreAdamW:
    """Create a GaLore optimizer with appropriate parameter groups.

    Automatically separates parameters into GaLore (large 2D tensors)
    and regular groups.

    Args:
        model: Model to optimize.
        lr: Learning rate.
        rank: Projection rank for GaLore params.
        update_proj_gap: Steps between projection updates.
        scale: Gradient scaling.
        weight_decay: Weight decay.
        target_modules: Module name patterns to apply GaLore to.

    Returns:
        Configured GaLoreAdamW optimizer.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    galore_params = []
    regular_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 2 and any(mod in name for mod in target_modules):
            galore_params.append(param)
        else:
            regular_params.append(param)

    param_groups = [
        {"params": regular_params, "use_galore": False},
        {"params": galore_params, "use_galore": True, "rank": rank,
         "update_proj_gap": update_proj_gap, "scale": scale},
    ]

    optimizer = GaLoreAdamW(
        param_groups, lr=lr, weight_decay=weight_decay,
        rank=rank, update_proj_gap=update_proj_gap, scale=scale,
    )

    logger.info(
        "GaLore optimizer: %d GaLore params, %d regular params, rank=%d",
        len(galore_params), len(regular_params), rank,
    )
    return optimizer
