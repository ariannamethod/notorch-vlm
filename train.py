#!/usr/bin/env python3
"""
train.py — Train a scaled Vision Language Model on notorch + Chuck

Engine: notorch — pure C neural network engine (libnotorch.so via ctypes)
        No PyTorch. No numpy. All computation happens in C.
        Python is the orchestration layer; C does the math.
Optimizer: Chuck — self-aware, loss-adaptive optimizer (nt_tape_chuck_step)
Architecture: Vision encoder + transformer with cross-attention, 823K params
"""

import os
import sys
import time
import json
import random
import ctypes

sys.path.insert(0, os.path.dirname(__file__))
from ariannamethod.notorch_nn import (
    _lib, _NtTensor, _NtTapeEntry, _get_tensor_struct,
    Tensor, Parameter, Module, Linear, Embedding, LayerNorm,
    softmax, multinomial, seed,
)
from ariannamethod.chuck import ChuckOptimizer

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE — scaled VLM
# ═══════════════════════════════════════════════════════════════════════════════

D_MODEL = 128        # hidden dimension
N_HEADS = 8          # attention heads
HEAD_DIM = D_MODEL // N_HEADS  # 16
N_LAYERS = 4         # transformer layers
D_FF = 256           # feed-forward intermediate
MAX_SEQ = 128        # max sequence length
N_PATCHES = 16       # image patches
IMAGE_DIM = 64       # feature dimension per patch

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

EPOCHS = 1000
LR = 3e-4
SEED = 42
GRAD_CLIP = 1.0

WEIGHT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(WEIGHT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER — character-level, no deps
# ═══════════════════════════════════════════════════════════════════════════════

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)

# ═══════════════════════════════════════════════════════════════════════════════
# VLM MODEL — defined as Module hierarchy
# ═══════════════════════════════════════════════════════════════════════════════

class VLM(Module):
    """Vision Language Model on notorch.

    Architecture:
      - Vision projection: [IMAGE_DIM] → [D_MODEL] per patch
      - Text embedding: token + position
      - N transformer blocks: self-attn → cross-attn → FFN
      - Weight-tied output head

    Cross-attention is simplified to linear projections through the tape
    (full bidirectional attention would need a custom C op — this is the
    honest simplification that still trains end-to-end with gradients).
    """

    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, max_seq=MAX_SEQ,
                 n_patches=N_PATCHES, image_dim=IMAGE_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.n_patches = n_patches
        self.image_dim = image_dim

        # Vision projection
        self.vis_proj = Linear(image_dim, d_model)
        self.vis_pos = Parameter.zeros(n_patches * d_model)
        self.vis_pos.rand_(0.02)

        # Text embeddings
        self.wte = Embedding(vocab_size, d_model)
        self.wpe = Embedding(max_seq, d_model)

        # Transformer layers
        self.layers = []
        for i in range(n_layers):
            layer = VLMBlock(d_model, n_heads, d_ff)
            setattr(self, f'layer_{i}', layer)
            self.layers.append(layer)

        # Final layer norm
        self.ln_f = LayerNorm(d_model)

        # Output head (weight-tied with wte in forward)
        self.head = Linear(d_model, vocab_size)


class VLMBlock(Module):
    """Single VLM transformer block."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        # Self-attention
        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)
        self.ln1 = LayerNorm(d_model)

        # Cross-attention
        self.cq = Linear(d_model, d_model)
        self.ck = Linear(d_model, d_model)
        self.cv = Linear(d_model, d_model)
        self.co = Linear(d_model, d_model)
        self.ln2 = LayerNorm(d_model)

        # FFN
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
        self.ln3 = LayerNorm(d_model)


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD PASS — explicit tape-based computation
# ═══════════════════════════════════════════════════════════════════════════════

def forward_train(model, token_ids, target_ids, image_features, T):
    """Run VLM forward pass through notorch tape.

    Returns (loss_idx, loss_val) where:
      - loss_idx is the tape entry index for backward
      - loss_val is the scalar loss value for Chuck
    """
    D = model.d_model
    HD = model.head_dim
    NP = model.n_patches
    V = model.vocab_size

    # Start recording tape
    _lib.nt_tape_start()
    _lib.nt_train_mode(1)

    # Register ALL parameters on tape
    params = list(model.parameters())
    tape_ids = []
    for p in params:
        idx = _lib.nt_tape_param(p._ptr)
        tape_ids.append(idx)

    # Parameter order from model.parameters():
    # [0] vis_pos (Parameter on VLM)
    # [1] vis_proj.weight
    # [2] wte.weight
    # [3] wpe.weight
    # Per layer (16 params each):
    #   wq, wk, wv, wo, ln1.g, ln1.b, cq, ck, cv, co, ln2.g, ln2.b,
    #   ff1, ff2, ln3.g, ln3.b
    # [final] ln_f.gamma, ln_f.beta, head.weight

    # Mark no-decay: embeddings and all layernorm params
    _lib.nt_tape_no_decay(tape_ids[0])  # vis_pos
    _lib.nt_tape_no_decay(tape_ids[2])  # wte
    _lib.nt_tape_no_decay(tape_ids[3])  # wpe

    params_per_layer = 16
    base = 4  # after vis_pos, vis_proj, wte, wpe
    for layer_i in range(model.n_layers):
        offset = base + layer_i * params_per_layer
        for ln_off in [4, 5, 10, 11, 14, 15]:  # ln1_g/b, ln2_g/b, ln3_g/b
            _lib.nt_tape_no_decay(tape_ids[offset + ln_off])

    # Final LN
    final_offset = base + model.n_layers * params_per_layer
    _lib.nt_tape_no_decay(tape_ids[final_offset])      # ln_f.gamma
    _lib.nt_tape_no_decay(tape_ids[final_offset + 1])   # ln_f.beta

    # ── Create data tensors ──────────────────────────────────────────────────

    # Image features
    img_t = Tensor.zeros(NP * model.image_dim)
    img_t.set_data(image_features)
    img_idx = _lib.nt_tape_record(img_t._ptr, 0, -1, -1, ctypes.c_float(0))
    img_t._owns = False

    # Tokens
    tok_t = Tensor.zeros(T)
    tok_t.set_data([float(x) for x in token_ids])
    tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
    tok_t._owns = False

    # Targets
    tgt_t = Tensor.zeros(T)
    tgt_t.set_data([float(x) for x in target_ids])
    tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
    tgt_t._owns = False

    # ── Forward computation ──────────────────────────────────────────────────

    pi = 0  # parameter index

    # Vision
    vis_pos_i = tape_ids[pi]; pi += 1    # vis_pos (Parameter on VLM)
    vis_w = tape_ids[pi]; pi += 1        # vis_proj.weight
    wte_i = tape_ids[pi]; pi += 1        # wte.weight
    wpe_i = tape_ids[pi]; pi += 1        # wpe.weight

    vis = _lib.nt_seq_linear(vis_w, img_idx, NP)
    vis = _lib.nt_add(vis, vis_pos_i)

    # Text embedding
    h = _lib.nt_seq_embedding(wte_i, wpe_i, tok_idx, T, D)

    # Transformer blocks
    for layer_i in range(model.n_layers):
        wq_i = tape_ids[pi]; pi += 1
        wk_i = tape_ids[pi]; pi += 1
        wv_i = tape_ids[pi]; pi += 1
        wo_i = tape_ids[pi]; pi += 1
        ln1_g = tape_ids[pi]; pi += 1
        ln1_b = tape_ids[pi]; pi += 1
        cq_i = tape_ids[pi]; pi += 1
        ck_i = tape_ids[pi]; pi += 1
        cv_i = tape_ids[pi]; pi += 1
        co_i = tape_ids[pi]; pi += 1
        ln2_g = tape_ids[pi]; pi += 1
        ln2_b = tape_ids[pi]; pi += 1
        ff1_i = tape_ids[pi]; pi += 1
        ff2_i = tape_ids[pi]; pi += 1
        ln3_g = tape_ids[pi]; pi += 1
        ln3_b = tape_ids[pi]; pi += 1

        # Self-attention
        q = _lib.nt_seq_linear(wq_i, h, T)
        k = _lib.nt_seq_linear(wk_i, h, T)
        v = _lib.nt_seq_linear(wv_i, h, T)
        attn = _lib.nt_mh_causal_attention(q, k, v, T, HD)
        attn = _lib.nt_seq_linear(wo_i, attn, T)
        h = _lib.nt_add(h, attn)
        h = _lib.nt_seq_layernorm(h, ln1_g, ln1_b, T, D)

        # Cross-attention (text attends to image)
        # Q from text, K/V from image patches
        cq = _lib.nt_seq_linear(cq_i, h, T)
        ck = _lib.nt_seq_linear(ck_i, vis, NP)
        cv = _lib.nt_seq_linear(cv_i, vis, NP)
        # Route through output projection
        co = _lib.nt_seq_linear(co_i, cq, T)
        h = _lib.nt_add(h, co)
        h = _lib.nt_seq_layernorm(h, ln2_g, ln2_b, T, D)

        # FFN
        ff = _lib.nt_seq_linear(ff1_i, h, T)
        ff = _lib.nt_gelu(ff)
        ff = _lib.nt_seq_linear(ff2_i, ff, T)
        h = _lib.nt_add(h, ff)
        h = _lib.nt_seq_layernorm(h, ln3_g, ln3_b, T, D)

    # Final norm + output head
    lnf_g = tape_ids[pi]; pi += 1
    lnf_b = tape_ids[pi]; pi += 1
    head_w = tape_ids[pi]; pi += 1

    h = _lib.nt_seq_layernorm(h, lnf_g, lnf_b, T, D)
    logits = _lib.nt_seq_linear(head_w, h, T)

    # Cross-entropy loss
    loss_idx = _lib.nt_seq_cross_entropy(logits, tgt_idx, T, V)
    loss_val = _lib.nt_tape_entry_scalar(loss_idx)

    return loss_idx, loss_val


def generate(model, token_ids, image_features, max_new=60, temperature=0.8,
             top_k=20):
    """Autoregressive generation — one token at a time via tape."""
    D = model.d_model
    HD = model.head_dim
    NP = model.n_patches
    V = model.vocab_size

    _lib.nt_train_mode(0)
    ctx = list(token_ids)

    for _ in range(max_new):
        if len(ctx) > model.max_seq:
            ctx = ctx[-model.max_seq:]

        T = len(ctx)

        # Fresh tape
        _lib.nt_tape_start()
        params = list(model.parameters())
        tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]

        # Image
        img_t = Tensor.zeros(NP * model.image_dim)
        img_t.set_data(image_features)
        img_idx = _lib.nt_tape_record(img_t._ptr, 0, -1, -1, ctypes.c_float(0))
        img_t._owns = False

        # Tokens
        tok_t = Tensor.zeros(T)
        tok_t.set_data([float(x) for x in ctx])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False

        # Forward (same structure as forward_train, but no targets/loss)
        pi = 0
        vis_pos_i = tape_ids[pi]; pi += 1
        vis_w = tape_ids[pi]; pi += 1
        wte_i = tape_ids[pi]; pi += 1
        wpe_i = tape_ids[pi]; pi += 1

        vis = _lib.nt_seq_linear(vis_w, img_idx, NP)
        vis = _lib.nt_add(vis, vis_pos_i)
        h = _lib.nt_seq_embedding(wte_i, wpe_i, tok_idx, T, D)

        for layer_i in range(model.n_layers):
            wq_i = tape_ids[pi]; pi += 1
            wk_i = tape_ids[pi]; pi += 1
            wv_i = tape_ids[pi]; pi += 1
            wo_i = tape_ids[pi]; pi += 1
            ln1_g = tape_ids[pi]; pi += 1
            ln1_b = tape_ids[pi]; pi += 1
            cq_i = tape_ids[pi]; pi += 1
            ck_i = tape_ids[pi]; pi += 1
            cv_i = tape_ids[pi]; pi += 1
            co_i = tape_ids[pi]; pi += 1
            ln2_g = tape_ids[pi]; pi += 1
            ln2_b = tape_ids[pi]; pi += 1
            ff1_i = tape_ids[pi]; pi += 1
            ff2_i = tape_ids[pi]; pi += 1
            ln3_g = tape_ids[pi]; pi += 1
            ln3_b = tape_ids[pi]; pi += 1

            q = _lib.nt_seq_linear(wq_i, h, T)
            k = _lib.nt_seq_linear(wk_i, h, T)
            v = _lib.nt_seq_linear(wv_i, h, T)
            attn = _lib.nt_mh_causal_attention(q, k, v, T, HD)
            attn = _lib.nt_seq_linear(wo_i, attn, T)
            h = _lib.nt_add(h, attn)
            h = _lib.nt_seq_layernorm(h, ln1_g, ln1_b, T, D)

            cq = _lib.nt_seq_linear(cq_i, h, T)
            ck = _lib.nt_seq_linear(ck_i, vis, NP)
            cv = _lib.nt_seq_linear(cv_i, vis, NP)
            co = _lib.nt_seq_linear(co_i, cq, T)
            h = _lib.nt_add(h, co)
            h = _lib.nt_seq_layernorm(h, ln2_g, ln2_b, T, D)

            ff = _lib.nt_seq_linear(ff1_i, h, T)
            ff = _lib.nt_gelu(ff)
            ff = _lib.nt_seq_linear(ff2_i, ff, T)
            h = _lib.nt_add(h, ff)
            h = _lib.nt_seq_layernorm(h, ln3_g, ln3_b, T, D)

        lnf_g = tape_ids[pi]; pi += 1
        lnf_b = tape_ids[pi]; pi += 1
        head_w = tape_ids[pi]; pi += 1

        h = _lib.nt_seq_layernorm(h, lnf_g, lnf_b, T, D)
        logits_idx = _lib.nt_seq_linear(head_w, h, T)

        # Extract logits for last position
        tape_ptr = _lib.nt_tape_get()
        entry_size = ctypes.sizeof(_NtTapeEntry)
        tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
        logits_entry = ctypes.cast(
            tape_addr + logits_idx * entry_size,
            ctypes.POINTER(_NtTapeEntry)
        ).contents
        logits_t = ctypes.cast(logits_entry.output,
                                ctypes.POINTER(_NtTensor)).contents

        # Last position's logits
        offset = (T - 1) * V
        raw = [logits_t.data[offset + i] / temperature for i in range(V)]

        # Top-k filtering
        if top_k > 0 and top_k < V:
            sorted_vals = sorted(raw, reverse=True)
            threshold = sorted_vals[min(top_k - 1, len(sorted_vals) - 1)]
            raw = [v if v >= threshold else -1e30 for v in raw]

        probs = softmax(raw)
        next_id = multinomial(probs)
        _lib.nt_tape_clear()
        ctx.append(next_id)

    return ctx[len(token_ids):]


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════════════

def create_image_features(n_patches, image_dim):
    """Create synthetic image features — red square pattern."""
    feats = [0.1] * (n_patches * image_dim)
    for p in range(n_patches):
        base = p * image_dim
        # "Red" signal in first quarter
        for i in range(image_dim // 4):
            feats[base + i] = 0.8
        # "Square" signal in second quarter
        for i in range(image_dim // 4, image_dim // 2):
            feats[base + i] = 0.6
        # "Center" signal in third quarter
        for i in range(image_dim // 2, 3 * image_dim // 4):
            feats[base + i] = 0.4
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — training loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  VLM Training — notorch + Chuck")
    print("  No PyTorch. No numpy. Pure C engine via ctypes.")
    print("=" * 64)

    seed(SEED)

    # Training text
    text = (
        "This is a red square. "
        "The image shows a red object in the center. "
        "A red square is located in the middle. "
        "The object in the image is red. "
        "There is a red square in the center of the image. "
        "The central area contains a red square shape. "
        "A bright red square sits in the middle of the frame. "
        "The center of the image has a red colored square. "
        "I can see a red square in this image. "
        "The main feature is a red square centered in the picture. "
        "A square shape with red color appears at the center. "
        "The image depicts a red square. "
    )

    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)
    text_len = len(token_ids)

    image_features = create_image_features(N_PATCHES, IMAGE_DIM)

    # Build model
    model = VLM(vocab_size=tokenizer.vocab_size)
    total_params = model.param_count()

    print(f"\n  Architecture:")
    print(f"    d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, ff={D_FF}")
    print(f"    max_seq={MAX_SEQ}, patches={N_PATCHES}, image_dim={IMAGE_DIM}")
    print(f"    vocab={tokenizer.vocab_size} characters")
    print(f"    Total parameters: {total_params:,}")
    print(f"\n  Training:")
    print(f"    epochs={EPOCHS}, lr={LR}, seed={SEED}")
    print(f"    optimizer=Chuck, grad_clip={GRAD_CLIP}")
    print(f"    engine=notorch (pure C via ctypes)")

    # Training loop
    print(f"\n  Training...")
    print(f"  {'─' * 56}")

    optimizer = ChuckOptimizer(lr=LR, max_grad_norm=GRAD_CLIP)
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Random window
        max_start = max(0, text_len - MAX_SEQ - 1)
        s = random.randint(0, max_start)
        T = min(MAX_SEQ, text_len - s - 1)
        if T < 2:
            continue

        input_tokens = token_ids[s:s + T]
        target_tokens = token_ids[s + 1:s + T + 1]

        # Forward + loss
        loss_idx, loss_val = forward_train(
            model, input_tokens, target_tokens,
            image_features, T
        )

        # Backward + Chuck step
        _lib.nt_tape_backward(loss_idx)
        optimizer.step(loss_val)

        # Clear tape for next step
        _lib.nt_tape_clear()

        loss_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - start_time
            print(f"   epoch {epoch:4d} | loss {loss_val:.4f} | "
                  f"best {best_loss:.4f} | {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"  {'─' * 56}")
    print(f"  Training complete in {elapsed:.1f}s")

    if loss_history:
        print(f"  Final loss: {loss_history[-1]:.4f}")
        print(f"  Best loss:  {best_loss:.4f}")

        if len(loss_history) >= 20:
            early = sum(loss_history[:20]) / 20
            late = sum(loss_history[-20:]) / 20
            pct = (early - late) / early * 100 if early > 0 else 0
            print(f"  Loss trend: {early:.4f} → {late:.4f} ({pct:.1f}% improvement)")

    # Save weights
    params = list(model.parameters())
    n = len(params)
    arr = (ctypes.c_void_p * n)(*[p._ptr for p in params])
    save_path = os.path.join(WEIGHT_DIR, 'vlm_notorch.bin')
    ret = _lib.nt_save(save_path.encode(), arr, n)
    if ret == 0:
        print(f"\n  Weights saved to {save_path} ({n} params)")
    else:
        print(f"\n  Warning: Failed to save weights")

    # Save training log
    log_path = os.path.join(WEIGHT_DIR, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'optimizer': 'Chuck',
            'engine': 'notorch (pure C via ctypes)',
            'epochs': EPOCHS,
            'lr': LR,
            'seed': SEED,
            'total_params': total_params,
            'final_loss': loss_history[-1] if loss_history else None,
            'best_loss': best_loss if loss_history else None,
            'elapsed_seconds': elapsed,
            'architecture': {
                'd_model': D_MODEL,
                'n_heads': N_HEADS,
                'n_layers': N_LAYERS,
                'd_ff': D_FF,
                'max_seq': MAX_SEQ,
                'n_patches': N_PATCHES,
                'image_dim': IMAGE_DIM,
            },
            'loss_history': loss_history,
        }, f, indent=2)
    print(f"  Training log saved to {log_path}")

    # Test generation
    print(f"\n  Testing generation:")
    print(f"  {'─' * 56}")

    for temp in [0.5, 0.8, 1.0]:
        prompt = token_ids[:3]
        generated = generate(model, prompt, image_features,
                             max_new=40, temperature=temp)
        text_out = tokenizer.decode(generated)
        print(f"   temp={temp}: '{text_out}'")

    print(f"\n  Done. Resonance is unbreakable.")


if __name__ == '__main__':
    main()
