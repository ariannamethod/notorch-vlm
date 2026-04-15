"""
notorch_nn.py — Python neural network API backed by libnotorch (ctypes)

Drop-in replacement for torch.nn / torch.nn.functional / torch.optim.
No PyTorch. No numpy. Just ctypes to the C library that started it all.

Based on ariannamethod/nanoGPT-notorch — the reference implementation.

Usage:
    from ariannamethod.notorch_nn import (
        _lib, Tensor, Parameter, Module, Linear, Embedding, LayerNorm,
        softmax, multinomial, seed,
    )
"""

import ctypes
import ctypes.util
import os
import math
import random

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD libnotorch
# ═══════════════════════════════════════════════════════════════════════════════

_dir = os.path.dirname(os.path.abspath(__file__))

# Try platform-specific extensions
for ext in ['.so', '.dylib', '.dll']:
    _libpath = os.path.join(_dir, f'libnotorch{ext}')
    if os.path.exists(_libpath):
        break
else:
    # Try building it
    _src = os.path.join(_dir, 'notorch.c')
    _libpath = os.path.join(_dir, 'libnotorch.so')
    if os.path.exists(_src):
        import subprocess
        subprocess.run(['cc', '-O2', '-std=c11', '-shared', '-fPIC',
                       '-o', _libpath, _src, '-lm'], check=True)

_lib = ctypes.CDLL(_libpath)

# ═══════════════════════════════════════════════════════════════════════════════
# C FUNCTION SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

# Tensor
_lib.nt_tensor_new.restype = ctypes.c_void_p
_lib.nt_tensor_new.argtypes = [ctypes.c_int]
_lib.nt_tensor_new2d.restype = ctypes.c_void_p
_lib.nt_tensor_new2d.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_tensor_free.argtypes = [ctypes.c_void_p]
_lib.nt_tensor_xavier.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.nt_tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.nt_tensor_rand.argtypes = [ctypes.c_void_p, ctypes.c_float]

# Tape
_lib.nt_tape_start.restype = None
_lib.nt_tape_clear.restype = None
_lib.nt_tape_param.restype = ctypes.c_int
_lib.nt_tape_param.argtypes = [ctypes.c_void_p]
_lib.nt_tape_no_decay.argtypes = [ctypes.c_int]
_lib.nt_tape_backward.argtypes = [ctypes.c_int]
_lib.nt_tape_clip_grads.restype = ctypes.c_float
_lib.nt_tape_clip_grads.argtypes = [ctypes.c_float]
_lib.nt_tape_chuck_step.argtypes = [ctypes.c_float, ctypes.c_float]
_lib.nt_tape_adam_step.argtypes = [ctypes.c_float]
_lib.nt_train_mode.argtypes = [ctypes.c_int]
_lib.nt_tape_get.restype = ctypes.c_void_p

# Tape helpers
_lib.nt_tape_entry_scalar.restype = ctypes.c_float
_lib.nt_tape_entry_scalar.argtypes = [ctypes.c_int]
_lib.nt_tape_reset_graph.restype = None

# Ops — all return tape entry index
_lib.nt_seq_embedding.restype = ctypes.c_int
_lib.nt_seq_embedding.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int]
_lib.nt_seq_linear.restype = ctypes.c_int
_lib.nt_seq_linear.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_seq_layernorm.restype = ctypes.c_int
_lib.nt_seq_layernorm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int]
_lib.nt_mh_causal_attention.restype = ctypes.c_int
_lib.nt_mh_causal_attention.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int]
_lib.nt_mh_cross_attention.restype = ctypes.c_int
_lib.nt_mh_cross_attention.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.nt_seq_cross_entropy.restype = ctypes.c_int
_lib.nt_seq_cross_entropy.argtypes = [ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int]
_lib.nt_add.restype = ctypes.c_int
_lib.nt_add.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_mul.restype = ctypes.c_int
_lib.nt_mul.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.nt_gelu.restype = ctypes.c_int
_lib.nt_gelu.argtypes = [ctypes.c_int]
_lib.nt_silu.restype = ctypes.c_int
_lib.nt_silu.argtypes = [ctypes.c_int]
_lib.nt_rope.restype = ctypes.c_int
_lib.nt_rope.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Record data on tape
_lib.nt_tape_record.restype = ctypes.c_int
_lib.nt_tape_record.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_float]

# Save/Load
_lib.nt_save.restype = ctypes.c_int
_lib.nt_save.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p),
                          ctypes.c_int]
_lib.nt_load.restype = ctypes.POINTER(ctypes.c_void_p)
_lib.nt_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

# Seed
_lib.nt_seed.argtypes = [ctypes.c_uint64]

# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR STRUCT (for data access)
# ═══════════════════════════════════════════════════════════════════════════════

class _NtTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int * 8),
        ("stride", ctypes.c_int * 8),
        ("len", ctypes.c_int),
        ("refcount", ctypes.c_int),
    ]


class _NtTapeEntry(ctypes.Structure):
    _fields_ = [
        ("output", ctypes.c_void_p),
        ("grad", ctypes.c_void_p),
        ("op", ctypes.c_int),
        ("parent1", ctypes.c_int),
        ("parent2", ctypes.c_int),
        ("parent3", ctypes.c_int),
        ("aux", ctypes.c_float),
        ("aux2", ctypes.c_float),
        ("aux3", ctypes.c_float),
        ("aux4", ctypes.c_float),
        ("is_param", ctypes.c_int),
        ("no_decay", ctypes.c_int),
    ]


def _get_tensor_struct(ptr):
    """Cast void* to _NtTensor*"""
    return ctypes.cast(ptr, ctypes.POINTER(_NtTensor)).contents


# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR / PARAMETER
# ═══════════════════════════════════════════════════════════════════════════════

class Tensor:
    """Wrapper around nt_tensor*"""

    def __init__(self, ptr, owns=True):
        self._ptr = ptr
        self._owns = owns

    @staticmethod
    def zeros(size):
        if isinstance(size, int):
            ptr = _lib.nt_tensor_new(size)
        elif len(size) == 1:
            ptr = _lib.nt_tensor_new(size[0])
        else:
            ptr = _lib.nt_tensor_new2d(size[0], size[1])
        return Tensor(ptr)

    @staticmethod
    def ones(size):
        t = Tensor.zeros(size) if isinstance(size, int) else Tensor.zeros(size)
        _lib.nt_tensor_fill(t._ptr, 1.0)
        return t

    @property
    def data_ptr(self):
        return _get_tensor_struct(self._ptr).data

    @property
    def numel(self):
        return _get_tensor_struct(self._ptr).len

    @property
    def shape(self):
        s = _get_tensor_struct(self._ptr)
        return tuple(s.shape[i] for i in range(s.ndim))

    def fill_(self, val):
        _lib.nt_tensor_fill(self._ptr, ctypes.c_float(val))
        return self

    def rand_(self, scale):
        _lib.nt_tensor_rand(self._ptr, ctypes.c_float(scale))
        return self

    def xavier_(self, fan_in, fan_out):
        _lib.nt_tensor_xavier(self._ptr, fan_in, fan_out)
        return self

    def set_data(self, flat_list):
        """Set tensor data from a flat list of floats"""
        s = _get_tensor_struct(self._ptr)
        n = min(len(flat_list), s.len)
        for i in range(n):
            s.data[i] = flat_list[i]

    def get_data(self):
        """Get tensor data as flat list"""
        s = _get_tensor_struct(self._ptr)
        return [s.data[i] for i in range(s.len)]

    def __del__(self):
        if self._owns and self._ptr:
            _lib.nt_tensor_free(self._ptr)


class Parameter(Tensor):
    """Trainable parameter — same as Tensor but marks ownership"""

    @staticmethod
    def zeros(size):
        if isinstance(size, int):
            ptr = _lib.nt_tensor_new(size)
        elif len(size) == 1:
            ptr = _lib.nt_tensor_new(size[0])
        else:
            ptr = _lib.nt_tensor_new2d(size[0], size[1])
        return Parameter(ptr)

    @staticmethod
    def ones(size):
        p = Parameter.zeros(size) if isinstance(size, int) else Parameter.zeros(size)
        _lib.nt_tensor_fill(p._ptr, 1.0)
        return p


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Module:
    """Base module — like nn.Module but backed by notorch"""

    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault('_parameters', {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault('_modules', {})[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        """Yield all parameters recursively"""
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def param_count(self):
        return sum(p.numel for p in self.parameters())

    def train(self, mode=True):
        self._training = mode
        _lib.nt_train_mode(1 if mode else 0)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)


class Linear(Module):
    """Drop-in for torch.nn.Linear"""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter.zeros((out_features, in_features))
        self.weight.xavier_(in_features, out_features)


class Embedding(Module):
    """Drop-in for torch.nn.Embedding"""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter.zeros((num_embeddings, embedding_dim))
        self.weight.rand_(0.02)


class LayerNorm(Module):
    """LayerNorm with gamma + beta"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = Parameter.ones(dim)
        self.beta = Parameter.zeros(dim)


class RMSNorm(Module):
    """RMSNorm with gamma"""
    def __init__(self, dim):
        super().__init__()
        self.weight = Parameter.ones(dim)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL API (replaces torch.nn.functional)
# ═══════════════════════════════════════════════════════════════════════════════

def softmax(logits_list):
    """Softmax over a list of floats"""
    mx = max(logits_list)
    exps = [math.exp(x - mx) for x in logits_list]
    s = sum(exps)
    return [e / s for e in exps]


def multinomial(probs):
    """Sample one index from probability distribution"""
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if cum >= r:
            return i
    return len(probs) - 1


# ═══════════════════════════════════════════════════════════════════════════════
# SEED
# ═══════════════════════════════════════════════════════════════════════════════

def seed(s):
    """Seed both notorch RNG and Python random"""
    _lib.nt_seed(ctypes.c_uint64(s))
    random.seed(s)


__all__ = [
    '_lib', '_NtTensor', '_NtTapeEntry', '_get_tensor_struct',
    'Tensor', 'Parameter', 'Module', 'Linear', 'Embedding',
    'LayerNorm', 'RMSNorm',
    'softmax', 'multinomial', 'seed',
]
