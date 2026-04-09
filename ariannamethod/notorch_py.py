"""
notorch_py.py — Python ctypes bindings for notorch (pure C neural network engine)

Usage:
    from ariannamethod.notorch_py import NotorchLib

    nt = NotorchLib()
    nt.tape_start()
    t = nt.tensor_new(10)
    idx = nt.tape_param(t)
    ...

Part of simple_vlm project (Arianna Method)
"""

import ctypes
import ctypes.util
import os
import numpy as np

# ── Load shared library ──────────────────────────────────────────────────────

_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, 'libnotorch.so')

if not os.path.exists(_lib_path):
    raise RuntimeError(
        f"libnotorch.so not found at {_lib_path}. "
        f"Build it: cc -std=c11 -O2 -fPIC -shared -o libnotorch.so notorch.c -lm"
    )

_lib = ctypes.CDLL(_lib_path)

# ── C types ──────────────────────────────────────────────────────────────────

_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_int_p = ctypes.POINTER(ctypes.c_int)


class _NtTensor(ctypes.Structure):
    """Mirror of nt_tensor from notorch.h"""
    _fields_ = [
        ("data", _c_float_p),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int * 8),     # NT_MAX_DIMS = 8
        ("stride", ctypes.c_int * 8),
        ("len", ctypes.c_int),
        ("refcount", ctypes.c_int),
    ]


_NtTensorPtr = ctypes.POINTER(_NtTensor)


# ── Function signatures ─────────────────────────────────────────────────────

# Tensor creation
_lib.nt_tensor_new.restype = _NtTensorPtr
_lib.nt_tensor_new.argtypes = [ctypes.c_int]

_lib.nt_tensor_new2d.restype = _NtTensorPtr
_lib.nt_tensor_new2d.argtypes = [ctypes.c_int, ctypes.c_int]

_lib.nt_tensor_free.restype = None
_lib.nt_tensor_free.argtypes = [_NtTensorPtr]

_lib.nt_tensor_fill.restype = None
_lib.nt_tensor_fill.argtypes = [_NtTensorPtr, ctypes.c_float]

_lib.nt_tensor_rand.restype = None
_lib.nt_tensor_rand.argtypes = [_NtTensorPtr, ctypes.c_float]

_lib.nt_tensor_xavier.restype = None
_lib.nt_tensor_xavier.argtypes = [_NtTensorPtr, ctypes.c_int, ctypes.c_int]

# Tape API
_lib.nt_tape_start.restype = None
_lib.nt_tape_start.argtypes = []

_lib.nt_tape_clear.restype = None
_lib.nt_tape_clear.argtypes = []

_lib.nt_tape_param.restype = ctypes.c_int
_lib.nt_tape_param.argtypes = [_NtTensorPtr]

_lib.nt_tape_no_decay.restype = None
_lib.nt_tape_no_decay.argtypes = [ctypes.c_int]

_lib.nt_tape_backward.restype = None
_lib.nt_tape_backward.argtypes = [ctypes.c_int]

_lib.nt_tape_adam_step.restype = None
_lib.nt_tape_adam_step.argtypes = [ctypes.c_float]

_lib.nt_tape_chuck_step.restype = None
_lib.nt_tape_chuck_step.argtypes = [ctypes.c_float, ctypes.c_float]

_lib.nt_tape_clip_grads.restype = ctypes.c_float
_lib.nt_tape_clip_grads.argtypes = [ctypes.c_float]

_lib.nt_tape_get.restype = ctypes.c_void_p
_lib.nt_tape_get.argtypes = []

# Training mode
_lib.nt_train_mode.restype = None
_lib.nt_train_mode.argtypes = [ctypes.c_int]

# Seed
_lib.nt_seed.restype = None
_lib.nt_seed.argtypes = [ctypes.c_uint64]

# Ops (return tape entry index)
_lib.nt_seq_embedding.restype = ctypes.c_int
_lib.nt_seq_embedding.argtypes = [ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.nt_seq_linear.restype = ctypes.c_int
_lib.nt_seq_linear.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.nt_seq_layernorm.restype = ctypes.c_int
_lib.nt_seq_layernorm.argtypes = [ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.nt_mh_causal_attention.restype = ctypes.c_int
_lib.nt_mh_causal_attention.argtypes = [ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int]

_lib.nt_seq_cross_entropy.restype = ctypes.c_int
_lib.nt_seq_cross_entropy.argtypes = [ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int]

_lib.nt_add.restype = ctypes.c_int
_lib.nt_add.argtypes = [ctypes.c_int, ctypes.c_int]

_lib.nt_gelu.restype = ctypes.c_int
_lib.nt_gelu.argtypes = [ctypes.c_int]

# Record raw entries (for injecting data tensors onto the tape)
_lib.nt_tape_record.restype = ctypes.c_int
_lib.nt_tape_record.argtypes = [_NtTensorPtr, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_float]

# Helper: read scalar from tape entry output
_lib.nt_tape_entry_scalar.restype = ctypes.c_float
_lib.nt_tape_entry_scalar.argtypes = [ctypes.c_int]

# Helper: reset computation graph (keep params)
_lib.nt_tape_reset_graph.restype = None
_lib.nt_tape_reset_graph.argtypes = []


# ── Python wrapper class ────────────────────────────────────────────────────

class NotorchLib:
    """High-level Python interface to the notorch C library."""

    def __init__(self):
        self._lib = _lib

    # -- Tensor ops --

    def tensor_new(self, length):
        return _lib.nt_tensor_new(length)

    def tensor_new2d(self, rows, cols):
        return _lib.nt_tensor_new2d(rows, cols)

    def tensor_fill(self, t, val):
        _lib.nt_tensor_fill(t, val)

    def tensor_rand(self, t, scale):
        _lib.nt_tensor_rand(t, scale)

    def tensor_xavier(self, t, fan_in, fan_out):
        _lib.nt_tensor_xavier(t, fan_in, fan_out)

    def tensor_free(self, t):
        _lib.nt_tensor_free(t)

    def tensor_to_numpy(self, t):
        """Copy tensor data to a numpy array."""
        n = t.contents.len
        arr = np.empty(n, dtype=np.float32)
        ctypes.memmove(arr.ctypes.data, t.contents.data, n * 4)
        return arr

    def tensor_from_numpy(self, arr):
        """Create a tensor from a numpy array (1D float32)."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        t = _lib.nt_tensor_new(len(arr))
        ctypes.memmove(t.contents.data, arr.ctypes.data, len(arr) * 4)
        return t

    def tensor_set_data(self, t, data):
        """Set tensor data from a list or numpy array."""
        if isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data, dtype=np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
        n = min(len(arr), t.contents.len)
        ctypes.memmove(t.contents.data, arr.ctypes.data, n * 4)

    def tensor_get_scalar(self, t):
        """Get first element of tensor as float."""
        return t.contents.data[0]

    # -- Tape ops --

    def seed(self, s):
        _lib.nt_seed(s)

    def tape_start(self):
        _lib.nt_tape_start()

    def tape_clear(self):
        _lib.nt_tape_clear()

    def tape_param(self, t):
        return _lib.nt_tape_param(t)

    def tape_no_decay(self, idx):
        _lib.nt_tape_no_decay(idx)

    def tape_backward(self, loss_idx):
        _lib.nt_tape_backward(loss_idx)

    def tape_clip_grads(self, max_norm):
        return _lib.nt_tape_clip_grads(max_norm)

    def tape_adam_step(self, lr):
        _lib.nt_tape_adam_step(lr)

    def tape_chuck_step(self, lr, loss_val):
        _lib.nt_tape_chuck_step(lr, loss_val)

    def train_mode(self, training):
        _lib.nt_train_mode(1 if training else 0)

    # -- Inject a data tensor onto the tape (no op, no grad) --

    def tape_push_data(self, t):
        """Record a data tensor onto the tape as NT_OP_NONE. Returns index."""
        return _lib.nt_tape_record(t, 0, -1, -1, 0.0)  # NT_OP_NONE = 0

    def tape_entry_scalar(self, idx):
        """Read the scalar value from a tape entry's output tensor."""
        return _lib.nt_tape_entry_scalar(idx)

    def tape_reset_graph(self):
        """Reset computation graph, keep params. Call between epochs."""
        _lib.nt_tape_reset_graph()

    # -- Neural network ops (all return tape entry indices) --

    def seq_embedding(self, wte_idx, wpe_idx, tokens_idx, T, D):
        return _lib.nt_seq_embedding(wte_idx, wpe_idx, tokens_idx, T, D)

    def seq_linear(self, w_idx, x_idx, T):
        return _lib.nt_seq_linear(w_idx, x_idx, T)

    def seq_layernorm(self, x_idx, gamma_idx, beta_idx, T, D):
        return _lib.nt_seq_layernorm(x_idx, gamma_idx, beta_idx, T, D)

    def mh_causal_attention(self, q_idx, k_idx, v_idx, T, head_dim):
        return _lib.nt_mh_causal_attention(q_idx, k_idx, v_idx, T, head_dim)

    def seq_cross_entropy(self, logits_idx, targets_idx, T, V):
        return _lib.nt_seq_cross_entropy(logits_idx, targets_idx, T, V)

    def add(self, a_idx, b_idx):
        return _lib.nt_add(a_idx, b_idx)

    def gelu(self, x_idx):
        return _lib.nt_gelu(x_idx)
