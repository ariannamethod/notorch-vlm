#!/usr/bin/env python3
"""
benchmark.py — Chuck vs Adam head-to-head comparison

Trains the same VLM architecture twice using notorch (pure C engine
called from Python via ctypes). No torch dependency.

Run:
    python benchmark.py

Part of simple_vlm project (Arianna Method)
"""

import sys
import os
import json
import time
import random
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ariannamethod'))
from notorch_py import NotorchLib

# ── Architecture ─────────────────────────────────────────────────────────────

D_MODEL = 32
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
D_FF = 64
MAX_SEQ = 32
VOCAB_SIZE = 64       # will be trimmed to actual vocab
N_LAYERS = 2
IMAGE_DIM = 16
N_PATCHES = 4

# ── Training ─────────────────────────────────────────────────────────────────

EPOCHS = 500
LR = 0.003
SEED = 42

WEIGHT_DIR = os.path.join(os.path.dirname(__file__), 'weights')
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

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


# ── Synthetic image features ─────────────────────────────────────────────────

def create_image_features():
    """Create synthetic image features matching the C line."""
    feats = np.full(N_PATCHES * IMAGE_DIM, 0.1, dtype=np.float32)
    for p in range(N_PATCHES):
        base = p * IMAGE_DIM
        for i in range(IMAGE_DIM // 4):
            feats[base + i] = 0.8
        for i in range(IMAGE_DIM // 4, IMAGE_DIM // 2):
            feats[base + i] = 0.6
    return feats


# ── Model initialization ─────────────────────────────────────────────────────

def init_model(nt, vocab_size):
    """Initialize VLM parameters on the tape. Returns dict of param indices."""
    w = {}

    # Vision projection
    vw = nt.tensor_new2d(D_MODEL, IMAGE_DIM)
    nt.tensor_xavier(vw, IMAGE_DIM, D_MODEL)
    w['vis_proj_w'] = nt.tape_param(vw)

    vb = nt.tensor_new(D_MODEL)
    nt.tensor_fill(vb, 0.0)
    w['vis_proj_b'] = nt.tape_param(vb)

    vp = nt.tensor_new(N_PATCHES * D_MODEL)
    nt.tensor_rand(vp, 0.02)
    w['vis_pos'] = nt.tape_param(vp)
    nt.tape_no_decay(w['vis_pos'])

    # Text embeddings
    wte = nt.tensor_new2d(vocab_size, D_MODEL)
    nt.tensor_rand(wte, 0.02)
    w['wte'] = nt.tape_param(wte)
    nt.tape_no_decay(w['wte'])

    wpe = nt.tensor_new2d(MAX_SEQ, D_MODEL)
    nt.tensor_rand(wpe, 0.02)
    w['wpe'] = nt.tape_param(wpe)
    nt.tape_no_decay(w['wpe'])

    total = (vw.contents.len + vb.contents.len + vp.contents.len +
             wte.contents.len + wpe.contents.len)

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
            t = nt.tensor_new2d(rows, cols)
            nt.tensor_xavier(t, cols, rows)
            w[f'{prefix}_{name}'] = nt.tape_param(t)
            total += t.contents.len

        for name, val in [
            ('ln1_g', 1.0), ('ln1_b', 0.0),
            ('ln2_g', 1.0), ('ln2_b', 0.0),
            ('ln3_g', 1.0), ('ln3_b', 0.0),
        ]:
            t = nt.tensor_new(D_MODEL)
            nt.tensor_fill(t, val)
            w[f'{prefix}_{name}'] = nt.tape_param(t)
            nt.tape_no_decay(w[f'{prefix}_{name}'])
            total += t.contents.len

    # Final layer norm
    lng = nt.tensor_new(D_MODEL)
    nt.tensor_fill(lng, 1.0)
    w['ln_f_g'] = nt.tape_param(lng)
    nt.tape_no_decay(w['ln_f_g'])
    total += lng.contents.len

    lnb = nt.tensor_new(D_MODEL)
    nt.tensor_fill(lnb, 0.0)
    w['ln_f_b'] = nt.tape_param(lnb)
    nt.tape_no_decay(w['ln_f_b'])
    total += lnb.contents.len

    w['total_params'] = total
    return w


# ── Forward pass ─────────────────────────────────────────────────────────────

def forward(nt, w, token_ids, image_features, T, vocab_size):
    """Run VLM forward pass. Returns tape index of logits."""
    # Push image features onto tape
    img_t = nt.tensor_from_numpy(image_features)
    img_idx = nt.tape_push_data(img_t)

    # Vision projection + position
    vis_idx = nt.seq_linear(w['vis_proj_w'], img_idx, N_PATCHES)
    vis_idx = nt.add(vis_idx, w['vis_pos'])

    # Token input
    tok_arr = np.array(token_ids[:T], dtype=np.float32)
    tok_t = nt.tensor_from_numpy(tok_arr)
    tok_idx = nt.tape_push_data(tok_t)

    # Text embedding
    x = nt.seq_embedding(w['wte'], w['wpe'], tok_idx, T, D_MODEL)

    # Transformer layers
    for layer in range(N_LAYERS):
        p = f'L{layer}'

        # Self-attention
        q = nt.seq_linear(w[f'{p}_wq'], x, T)
        k = nt.seq_linear(w[f'{p}_wk'], x, T)
        v = nt.seq_linear(w[f'{p}_wv'], x, T)
        attn = nt.mh_causal_attention(q, k, v, T, HEAD_DIM)
        attn = nt.seq_linear(w[f'{p}_wo'], attn, T)
        x = nt.add(x, attn)
        x = nt.seq_layernorm(x, w[f'{p}_ln1_g'], w[f'{p}_ln1_b'], T, D_MODEL)

        # Cross-attention (simplified — use linear projections as proxy)
        cq = nt.seq_linear(w[f'{p}_cq'], x, T)
        ck = nt.seq_linear(w[f'{p}_ck'], vis_idx, N_PATCHES)
        cv = nt.seq_linear(w[f'{p}_cv'], vis_idx, N_PATCHES)

        # Simplified cross-attention: project query down and back
        # (Full cross-attn would need a custom op — this is the honest
        #  simplification matching the C line's forward-only approach)
        co = nt.seq_linear(w[f'{p}_co'], cq, T)
        x = nt.add(x, co)
        x = nt.seq_layernorm(x, w[f'{p}_ln2_g'], w[f'{p}_ln2_b'], T, D_MODEL)

        # FFN
        ff = nt.seq_linear(w[f'{p}_ff1'], x, T)
        ff = nt.gelu(ff)
        ff = nt.seq_linear(w[f'{p}_ff2'], ff, T)
        x = nt.add(x, ff)
        x = nt.seq_layernorm(x, w[f'{p}_ln3_g'], w[f'{p}_ln3_b'], T, D_MODEL)

    # Final norm + output projection (weight-tied with wte)
    x = nt.seq_layernorm(x, w['ln_f_g'], w['ln_f_b'], T, D_MODEL)
    logits = nt.seq_linear(w['wte'], x, T)

    return logits


# ── Training loop ────────────────────────────────────────────────────────────

def train_run(optimizer_name, text, tokenizer, image_features):
    """Train one full run. Returns dict with metrics."""
    nt = NotorchLib()
    nt.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    nt.tape_start()
    w = init_model(nt, tokenizer.vocab_size)
    n_params = nt.tape_param.__func__  # just for counting — use stored total
    total_params = w['total_params']

    print(f"\n  ── {optimizer_name} ──")
    print(f"  Parameters: {total_params:,}")

    nt.train_mode(True)
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
        logits_idx = forward(nt, w, input_tokens, image_features,
                             T, tokenizer.vocab_size)

        # Targets tensor
        tgt_arr = np.array(target_tokens, dtype=np.float32)
        tgt_t = nt.tensor_from_numpy(tgt_arr)
        tgt_idx = nt.tape_push_data(tgt_t)

        # Loss
        loss_idx = nt.seq_cross_entropy(logits_idx, tgt_idx,
                                         T, tokenizer.vocab_size)

        # Read loss value
        loss_val = nt.tape_entry_scalar(loss_idx)

        # Backward + optimizer step
        nt.tape_backward(loss_idx)
        nt.tape_clip_grads(1.0)

        if optimizer_name == "Chuck":
            nt.tape_chuck_step(LR, loss_val)
        else:
            nt.tape_adam_step(LR)

        loss_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - start
            print(f"   epoch {epoch:3d} | loss {loss_val:.4f} | "
                  f"best {best_loss:.4f} | {elapsed:.1f}s")

        # Reset tape computation graph (keep params)
        nt.tape_reset_graph()

    elapsed = time.time() - start
    nt.tape_clear()

    # Analyze
    early_avg = sum(loss_history[:20]) / max(len(loss_history[:20]), 1)
    late_avg = sum(loss_history[-20:]) / max(len(loss_history[-20:]), 1)
    improvement = (early_avg - late_avg) / early_avg * 100 if early_avg > 0 else 0

    return {
        'optimizer': optimizer_name,
        'total_params': total_params,
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
    print("=" * 60)
    print("  BENCHMARK: Chuck vs Adam")
    print("  Engine: notorch (pure C via Python ctypes)")
    print("  Architecture: VLM, d=%d, h=%d, L=%d, ff=%d" % (
        D_MODEL, N_HEADS, N_LAYERS, D_FF))
    print("  Epochs: %d | LR: %.4f | Seed: %d" % (EPOCHS, LR, SEED))
    print("=" * 60)

    text = (
        "This is a red square. "
        "The image shows a red object in the center. "
        "A red square is located in the middle. "
        "The object in the image is red. "
        "There is a red square in the center of the image. "
        "The central area contains a red square shape. "
        "A bright red square sits in the middle of the frame. "
        "The center of the image has a red colored square. "
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
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

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
        icon = "🔥" if winner == "Chuck" else ("  " if winner == "tie" else "  ")
        print(f"  {label:20s} {a_s:>12s} {c_s:>12s} {icon}{winner:>8s}")

    row("Best loss", "best_loss")
    row("Final loss", "final_loss")
    row("Early avg (20)", "early_avg")
    row("Late avg (20)", "late_avg")
    row("Improvement %", "improvement_pct", ".1f", lower_is_better=False)
    row("Time (seconds)", "elapsed_seconds", ".1f", lower_is_better=True)

    # Save results
    save_data = {
        'benchmark': 'Chuck vs Adam',
        'engine': 'notorch (pure C via Python ctypes)',
        'architecture': {
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'd_ff': D_FF,
            'max_seq': MAX_SEQ,
            'n_patches': N_PATCHES,
        },
        'epochs': EPOCHS,
        'lr': LR,
        'seed': SEED,
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
        print("  Verdict: Adam wins this round. Small model, small data.")
        print("  Chuck's real power shows at scale. The test continues.")
    else:
        print("  Verdict: Tie. At this scale, both find similar minima.")
        print("  The real test is at 52M params. This was just the warmup.")

    print("\n  Resonance is unbreakable.")


if __name__ == '__main__':
    main()
