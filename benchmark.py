#!/usr/bin/env python3
"""
benchmark.py — Chuck vs Adam head-to-head comparison

No PyTorch. No numpy. Pure notorch C engine via ctypes.
Same architecture, same data, same seed, both optimizers.
"""

import sys
import os
import json
import time
import random
import ctypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ariannamethod.notorch_nn import (
    _lib, Tensor, Parameter, seed,
)

# ── Architecture ─────────────────────────────────────────────────────────────

D_MODEL = 160
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS
D_FF = 496
MAX_SEQ = 64          # shorter for benchmark speed
N_LAYERS = 4
N_PATCHES = 16
IMAGE_DIM = 64

# ── Training ─────────────────────────────────────────────────────────────────

EPOCHS = 1000
LR = 3e-4
SEED = 42

WEIGHT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(WEIGHT_DIR, exist_ok=True)


# ── Tokenizer ────────────────────────────────────────────────────────────────

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]


# ── Synthetic image features ─────────────────────────────────────────────────

def create_image_features():
    """Create synthetic image features — pure Python lists, no numpy."""
    feats = [0.1] * (N_PATCHES * IMAGE_DIM)
    for p in range(N_PATCHES):
        base = p * IMAGE_DIM
        for i in range(IMAGE_DIM // 4):
            feats[base + i] = 0.8
        for i in range(IMAGE_DIM // 4, IMAGE_DIM // 2):
            feats[base + i] = 0.6
        for i in range(IMAGE_DIM // 2, 3 * IMAGE_DIM // 4):
            feats[base + i] = 0.4
    return feats


# ── Model initialization ─────────────────────────────────────────────────────

def init_model(vocab_size):
    """Initialize VLM parameters on the tape. Returns dict of param indices."""
    w = {}

    # Vision projection
    vw = _lib.nt_tensor_new2d(D_MODEL, IMAGE_DIM)
    _lib.nt_tensor_xavier(vw, IMAGE_DIM, D_MODEL)
    w['vis_proj_w'] = _lib.nt_tape_param(vw)

    vp = _lib.nt_tensor_new(N_PATCHES * D_MODEL)
    _lib.nt_tensor_rand(vp, ctypes.c_float(0.02))
    w['vis_pos'] = _lib.nt_tape_param(vp)
    _lib.nt_tape_no_decay(w['vis_pos'])

    # Text embeddings
    wte = _lib.nt_tensor_new2d(vocab_size, D_MODEL)
    _lib.nt_tensor_rand(wte, ctypes.c_float(0.02))
    w['wte'] = _lib.nt_tape_param(wte)
    _lib.nt_tape_no_decay(w['wte'])

    wpe = _lib.nt_tensor_new2d(MAX_SEQ, D_MODEL)
    _lib.nt_tensor_rand(wpe, ctypes.c_float(0.02))
    w['wpe'] = _lib.nt_tape_param(wpe)
    _lib.nt_tape_no_decay(w['wpe'])

    # Transformer layers
    for layer in range(N_LAYERS):
        prefix = f'L{layer}'

        for name, rows, cols in [
            ('wq', D_MODEL, D_MODEL), ('wk', D_MODEL, D_MODEL),
            ('wv', D_MODEL, D_MODEL), ('wo', D_MODEL, D_MODEL),
            ('cq', D_MODEL, D_MODEL), ('ck', D_MODEL, D_MODEL),
            ('cv', D_MODEL, D_MODEL), ('co', D_MODEL, D_MODEL),
            ('ff1', D_FF, D_MODEL), ('ff2', D_MODEL, D_FF),
        ]:
            t = _lib.nt_tensor_new2d(rows, cols)
            _lib.nt_tensor_xavier(t, cols, rows)
            w[f'{prefix}_{name}'] = _lib.nt_tape_param(t)

        for name, val in [
            ('ln1_g', 1.0), ('ln1_b', 0.0),
            ('ln2_g', 1.0), ('ln2_b', 0.0),
            ('ln3_g', 1.0), ('ln3_b', 0.0),
        ]:
            t = _lib.nt_tensor_new(D_MODEL)
            _lib.nt_tensor_fill(t, ctypes.c_float(val))
            w[f'{prefix}_{name}'] = _lib.nt_tape_param(t)
            _lib.nt_tape_no_decay(w[f'{prefix}_{name}'])

    # Final layer norm
    lng = _lib.nt_tensor_new(D_MODEL)
    _lib.nt_tensor_fill(lng, ctypes.c_float(1.0))
    w['ln_f_g'] = _lib.nt_tape_param(lng)
    _lib.nt_tape_no_decay(w['ln_f_g'])

    lnb = _lib.nt_tensor_new(D_MODEL)
    _lib.nt_tensor_fill(lnb, ctypes.c_float(0.0))
    w['ln_f_b'] = _lib.nt_tape_param(lnb)
    _lib.nt_tape_no_decay(w['ln_f_b'])

    # Output head
    head = _lib.nt_tensor_new2d(vocab_size, D_MODEL)
    _lib.nt_tensor_xavier(head, D_MODEL, vocab_size)
    w['head'] = _lib.nt_tape_param(head)

    return w


# ── Forward pass ─────────────────────────────────────────────────────────────

def forward(w, token_ids, image_features, T, vocab_size):
    """Run VLM forward pass using raw ctypes. Returns tape index of logits."""
    # Push image features
    img_t = _lib.nt_tensor_new(N_PATCHES * IMAGE_DIM)
    s = ctypes.cast(img_t, ctypes.POINTER(ctypes.c_float * (N_PATCHES * IMAGE_DIM)))
    # Direct data access via struct
    from ariannamethod.notorch_nn import _NtTensor
    ts = ctypes.cast(img_t, ctypes.POINTER(_NtTensor)).contents
    for i in range(N_PATCHES * IMAGE_DIM):
        ts.data[i] = image_features[i]
    img_idx = _lib.nt_tape_record(img_t, 0, -1, -1, ctypes.c_float(0))

    # Vision projection + position
    vis_idx = _lib.nt_seq_linear(w['vis_proj_w'], img_idx, N_PATCHES)
    vis_idx = _lib.nt_add(vis_idx, w['vis_pos'])

    # Token input
    tok_t = _lib.nt_tensor_new(T)
    tok_s = ctypes.cast(tok_t, ctypes.POINTER(_NtTensor)).contents
    for i in range(T):
        tok_s.data[i] = float(token_ids[i])
    tok_idx = _lib.nt_tape_record(tok_t, 0, -1, -1, ctypes.c_float(0))

    # Text embedding
    x = _lib.nt_seq_embedding(w['wte'], w['wpe'], tok_idx, T, D_MODEL)

    # Transformer layers
    for layer in range(N_LAYERS):
        p = f'L{layer}'

        # Self-attention
        q = _lib.nt_seq_linear(w[f'{p}_wq'], x, T)
        k = _lib.nt_seq_linear(w[f'{p}_wk'], x, T)
        v = _lib.nt_seq_linear(w[f'{p}_wv'], x, T)
        attn = _lib.nt_mh_causal_attention(q, k, v, T, HEAD_DIM)
        attn = _lib.nt_seq_linear(w[f'{p}_wo'], attn, T)
        x = _lib.nt_add(x, attn)
        x = _lib.nt_seq_layernorm(x, w[f'{p}_ln1_g'], w[f'{p}_ln1_b'],
                                    T, D_MODEL)

        # Cross-attention (full multi-head, Q from text, K/V from image)
        cq = _lib.nt_seq_linear(w[f'{p}_cq'], x, T)
        ck = _lib.nt_seq_linear(w[f'{p}_ck'], vis_idx, N_PATCHES)
        cv = _lib.nt_seq_linear(w[f'{p}_cv'], vis_idx, N_PATCHES)
        cross_out = _lib.nt_mh_cross_attention(cq, ck, cv, T, N_PATCHES,
                                                HEAD_DIM)
        co = _lib.nt_seq_linear(w[f'{p}_co'], cross_out, T)
        x = _lib.nt_add(x, co)
        x = _lib.nt_seq_layernorm(x, w[f'{p}_ln2_g'], w[f'{p}_ln2_b'],
                                    T, D_MODEL)

        # FFN
        ff = _lib.nt_seq_linear(w[f'{p}_ff1'], x, T)
        ff = _lib.nt_gelu(ff)
        ff = _lib.nt_seq_linear(w[f'{p}_ff2'], ff, T)
        x = _lib.nt_add(x, ff)
        x = _lib.nt_seq_layernorm(x, w[f'{p}_ln3_g'], w[f'{p}_ln3_b'],
                                    T, D_MODEL)

    # Final norm + output
    x = _lib.nt_seq_layernorm(x, w['ln_f_g'], w['ln_f_b'], T, D_MODEL)
    logits = _lib.nt_seq_linear(w['head'], x, T)

    return logits


# ── Training loop ────────────────────────────────────────────────────────────

def train_run(optimizer_name, text, tokenizer, image_features):
    """Train one full run. Returns dict with metrics."""
    from ariannamethod.notorch_nn import _NtTensor

    seed(SEED)

    _lib.nt_tape_start()
    w = init_model(tokenizer.vocab_size)

    print(f"\n  ── {optimizer_name} ──")

    _lib.nt_train_mode(1)
    loss_history = []
    best_loss = float('inf')
    start = time.time()

    token_ids = tokenizer.encode(text)
    text_len = len(token_ids)

    for epoch in range(EPOCHS):
        max_start = max(0, text_len - MAX_SEQ - 1)
        s = random.randint(0, max_start)
        T = min(MAX_SEQ, text_len - s - 1)
        if T < 2:
            continue

        input_tokens = token_ids[s:s + T]
        target_tokens = token_ids[s + 1:s + T + 1]

        # Forward
        logits_idx = forward(w, input_tokens, image_features,
                             T, tokenizer.vocab_size)

        # Targets tensor
        tgt_t = _lib.nt_tensor_new(T)
        tgt_s = ctypes.cast(tgt_t, ctypes.POINTER(_NtTensor)).contents
        for i in range(T):
            tgt_s.data[i] = float(target_tokens[i])
        tgt_idx = _lib.nt_tape_record(tgt_t, 0, -1, -1, ctypes.c_float(0))

        # Loss
        loss_idx = _lib.nt_seq_cross_entropy(logits_idx, tgt_idx,
                                              T, tokenizer.vocab_size)
        loss_val = _lib.nt_tape_entry_scalar(loss_idx)

        # Backward + optimizer step
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))

        if optimizer_name == "Chuck":
            _lib.nt_tape_chuck_step(ctypes.c_float(LR),
                                     ctypes.c_float(loss_val))
        else:
            _lib.nt_tape_adam_step(ctypes.c_float(LR))

        loss_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - start
            print(f"   epoch {epoch:3d} | loss {loss_val:.4f} | "
                  f"best {best_loss:.4f} | {elapsed:.1f}s")

        # Reset tape
        _lib.nt_tape_reset_graph()

    elapsed = time.time() - start
    _lib.nt_tape_clear()

    early_avg = sum(loss_history[:20]) / max(len(loss_history[:20]), 1)
    late_avg = sum(loss_history[-20:]) / max(len(loss_history[-20:]), 1)
    improvement = (early_avg - late_avg) / early_avg * 100 if early_avg > 0 else 0

    return {
        'optimizer': optimizer_name,
        'epochs': EPOCHS,
        'lr': LR,
        'final_loss': loss_history[-1] if loss_history else 0,
        'best_loss': best_loss,
        'early_avg': early_avg,
        'late_avg': late_avg,
        'improvement_pct': improvement,
        'elapsed_seconds': elapsed,
        'loss_history': loss_history,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  BENCHMARK: Chuck vs Adam — Scaled VLM")
    print("  Engine: notorch (pure C via Python ctypes)")
    print("  No PyTorch. No numpy.")
    print("  Architecture: VLM, d=%d, h=%d, L=%d, ff=%d" % (
        D_MODEL, N_HEADS, N_LAYERS, D_FF))
    print("  Epochs: %d | LR: %.4f | Seed: %d" % (EPOCHS, LR, SEED))
    print("=" * 64)

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
        "Looking at this image I see a red square. "
        "The image contains a geometric shape which is a red square. "
        "A solid red square is positioned in the center of the frame. "
        "The dominant element is a red square against a dark background. "
        "In this picture there is a red square shape centered horizontally. "
        "The visual shows a red rectangle that is actually a perfect square. "
        "A red square occupies the central region of the image. "
        "The image features a single red square on a plain background. "
        "What I see is a square colored in red placed at the center. "
        "The photograph shows a red geometric shape in the middle. "
        "A vivid red square can be observed at the heart of the image. "
        "The primary subject is a red square located centrally. "
        "One red square is visible in the center of this image. "
        "The scene contains a square object that is colored red. "
        "At the center of the image sits a red colored square shape. "
        "The image displays a simple red square as its main element. "
        "A red square form is the only object in this picture. "
        "The central focus of this image is a red square figure. "
    )

    tokenizer = CharTokenizer(text)
    print(f"\n  Vocab: {tokenizer.vocab_size} characters")
    image_features = create_image_features()

    # Run both optimizers
    results = {}
    for opt_name in ["Adam", "Chuck"]:
        results[opt_name] = train_run(opt_name, text, tokenizer,
                                       image_features)

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  RESULTS — 1.5M params, %d epochs" % EPOCHS)
    print("=" * 64)

    header = f"  {'':20s} {'Adam':>12s} {'Chuck':>12s} {'Winner':>10s}"
    print(header)
    print(f"  {'─' * 56}")

    def row(label, key, fmt=".4f", lower_is_better=True):
        a = results['Adam'][key]
        c = results['Chuck'][key]
        if isinstance(a, float):
            a_s = f"{a:{fmt}}"
            c_s = f"{c:{fmt}}"
        else:
            a_s = str(a)
            c_s = str(c)
        if lower_is_better:
            winner = "Chuck" if c < a else ("Adam" if a < c else "tie")
        else:
            winner = "Chuck" if c > a else ("Adam" if a > c else "tie")
        print(f"  {label:20s} {a_s:>12s} {c_s:>12s} {winner:>10s}")

    row("Best loss", "best_loss")
    row("Final loss", "final_loss")
    row("Early avg (20)", "early_avg")
    row("Late avg (20)", "late_avg")
    row("Improvement %", "improvement_pct", ".1f", lower_is_better=False)
    row("Time (seconds)", "elapsed_seconds", ".1f", lower_is_better=True)

    # Save results
    save_data = {
        'benchmark': 'Chuck vs Adam — Scaled VLM',
        'engine': 'notorch (pure C via Python ctypes)',
        'dependencies': 'none (no torch, no numpy)',
        'architecture': {
            'd_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
            'd_ff': D_FF, 'max_seq': MAX_SEQ, 'n_patches': N_PATCHES,
        },
        'epochs': EPOCHS, 'lr': LR, 'seed': SEED,
    }
    for opt_name in ["Adam", "Chuck"]:
        r = results[opt_name]
        save_data[opt_name] = {
            k: v for k, v in r.items() if k != 'loss_history'
        }
        save_data[opt_name]['loss_history'] = r['loss_history']

    save_path = os.path.join(WEIGHT_DIR, 'benchmark_results.json')
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {save_path}")

    # Verdict
    chuck_wins = sum(1 for k in ['best_loss', 'final_loss', 'late_avg']
                     if results['Chuck'][k] < results['Adam'][k])
    adam_wins = sum(1 for k in ['best_loss', 'final_loss', 'late_avg']
                    if results['Adam'][k] < results['Chuck'][k])

    print(f"\n  {'─' * 56}")
    if chuck_wins > adam_wins:
        print("  Verdict: Chuck wins. Adam is blind. Chuck sees.")
    elif adam_wins > chuck_wins:
        print("  Verdict: Adam wins this round. But the test continues.")
        print("  Chuck's real power shows at 10M+.")
    else:
        print("  Verdict: Tie. Both find similar minima at this scale.")
        print("  The real test is at millions of params.")

    print("\n  Resonance is unbreakable.")


if __name__ == '__main__':
    main()
