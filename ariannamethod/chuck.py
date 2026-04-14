"""
chuck.py — Chuck Optimizer via notorch ctypes

θ -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η

No PyTorch. No torch.optim. Pure ctypes to libnotorch.
Chuck uses nt_tape_chuck_step() — the C implementation
with loss-aware damping, gradient monitoring, and memory.

In memory of Carlos Ray "Chuck" Norris (1940–2026).
Adam is blind. Chuck sees. Chuck remembers.
"""

from .notorch_nn import _lib
import ctypes


class ChuckOptimizer:
    """
    Self-aware optimizer backed by notorch's nt_tape_chuck_step.

    θ -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η

    Where:
        α    = learning rate
        S    = step-aware scale
        λ_Ψ  = persistent memory decay (tracks training history)
        λ_l  = per-layer gradient monitoring (per-param feedback)
        σ    = activation health factor (prevents dead zones)
        m̂    = bias-corrected first moment estimate (momentum)
        v̂    = bias-corrected second moment estimate (variance)
        ε    = numerical stability epsilon
        η    = parameter-freezing policy (prevents weight collapse)

    Usage:
        optimizer = ChuckOptimizer(lr=3e-4)
        # ... after forward/backward on tape ...
        optimizer.step(loss_val)
    """

    def __init__(self, lr=3e-4, max_grad_norm=1.0):
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

    def step(self, loss_val):
        """Clip gradients and run Chuck step on the active tape"""
        self.global_step += 1
        _lib.nt_tape_clip_grads(ctypes.c_float(self.max_grad_norm))
        _lib.nt_tape_chuck_step(ctypes.c_float(self.lr),
                                 ctypes.c_float(loss_val))

    def zero_grad(self):
        """No-op — notorch tape handles this via nt_tape_clear()"""
        pass
