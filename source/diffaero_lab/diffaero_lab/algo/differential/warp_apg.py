# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native APG optimizer.

WarpAPG owns the warp.optim.SGD optimizer and orchestrates the
wp.Tape()-based forward-backward-update cycle for differentiable
policy learning.

The optimizer operates on the flat params array from WarpDroneRollout.
After each train step, params.grad contains the accumulated gradient
from tape.backward(loss), which warp.optim.SGD consumes.
"""

from __future__ import annotations

import warp as wp
import warp.optim

from diffaero_lab.uav.differential.rollout import WarpDroneRollout


class WarpAPG:
    """Warp-native APG optimizer.

    Wraps warp.optim.SGD over the rollout's MLP parameters.
    Provides train_step() for the forward+backward+update cycle
    and export_policy_params() for IsaacLab evaluation.

    Usage:
        apg = WarpAPG(rollout, lr=1e-3)
        for i in range(iterations):
            metrics = apg.train_step(init_body_q, init_body_qd, init_obs)
            print(f"Iter {i}: loss={metrics['loss']:.4f}, grad_norm={metrics['grad_norm']:.4f}")
    """

    def __init__(
        self,
        rollout: WarpDroneRollout,
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        nesterov: bool = False,
        momentum: float = 0.0,
    ):
        """Initialize Warp-native APG optimizer.

        Args:
            rollout: WarpDroneRollout instance with requires_grad=True params.
            lr: Learning rate for SGD.
            max_grad_norm: Max gradient norm for clipping (applied pre-optimizer).
            nesterov: Whether to use Nesterov momentum.
            momentum: Momentum coefficient.
        """
        self._rollout = rollout
        self._lr = lr
        self._max_grad_norm = max_grad_norm

        # Warp SGD optimizer operating on the flat params array
        self._optimizer = warp.optim.SGD(
            [rollout.params],
            lr=lr,
            nesterov=nesterov,
            momentum=momentum,
        )

        # CUDA graph capture (faster subsequent iterations)
        self._graph: wp.Capture | None = None
        self._graph_captured = False
        self._last_grad_norm = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> dict[str, float]:
        """Execute one forward+backward+update cycle."""
        self._train_step_impl()
        wp.synchronize()
        loss_val = float(self._rollout.loss.numpy()[0])
        grad_norm = self._compute_grad_norm()
        return {"loss": loss_val, "grad_norm": self._last_grad_norm}

    def _train_step_impl(self) -> None:
        """Core forward+backward+optimizer logic."""
        tape = wp.Tape()
        with tape:
            self._rollout.forward()
        tape.backward(self._rollout.loss)

        # Compute grad norm before clipping/step (which zeros grads)
        self._last_grad_norm = self._compute_grad_norm()

        # Gradient clipping
        if self._max_grad_norm is not None and self._max_grad_norm > 0:
            grad = self._rollout.params.grad
            if grad is not None:
                self._clip_grad_norm(grad, self._max_grad_norm)

        # Optimizer step
        self._optimizer.step([self._rollout.params.grad])
        tape.zero()

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    def _compute_grad_norm(self) -> float:
        """Compute L2 norm of grads across all params."""
        grad = self._rollout.params.grad
        if grad is None:
            return 0.0
        grad_np = grad.numpy()
        total = float(sum(g * g for g in grad_np))
        return total**0.5

    @staticmethod
    def _clip_grad_norm(grad: wp.array, max_norm: float) -> None:
        """Clip gradient L2 norm in-place."""
        grad_np = grad.numpy()
        norm = float(sum(g * g for g in grad_np)) ** 0.5
        if norm > max_norm:
            scale = max_norm / norm
            grad_np *= scale
            grad.assign(grad_np)

    # ------------------------------------------------------------------
    # Policy export
    # ------------------------------------------------------------------

    def export_policy_params(self) -> wp.array:
        """Return a copy of the current MLP parameters.

        Returns:
            Flat wp.array (numpy-compatible) of all weights and biases.
            Layout matches init_mlp_params: [W0, b0, W1, b1, ...].
        """
        return self._rollout.params.numpy().copy()

    def export_policy_torch(self) -> dict[str, object]:
        """Export MLP weights to PyTorch state_dict format.

        Returns:
            Dict mapping layer names to torch.Tensor weight/bias.
            Compatible with torch.nn.Sequential loading via load_state_dict.

        Raises:
            ImportError: If torch is not available.
        """
        import torch

        layer_dims = self._rollout.layer_dims
        params_np = self._rollout.params.numpy()

        state_dict: dict[str, object] = {}
        offset = 0
        num_layers = len(layer_dims) - 1
        for idx in range(num_layers):
            in_d = layer_dims[idx]
            out_d = layer_dims[idx + 1]
            w_size = out_d * in_d

            w = params_np[offset : offset + w_size].reshape(out_d, in_d)
            offset += w_size
            b = params_np[offset : offset + out_d]
            offset += out_d

            state_dict[f"network.{2 * idx}.weight"] = torch.tensor(w.copy())
            state_dict[f"network.{2 * idx}.bias"] = torch.tensor(b.copy())

        return state_dict
