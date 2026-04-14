#!/usr/bin/env python3
"""
train.py — Train a ~20K parameter VLM prototype
Python line: torch tensors + Chuck optimizer
Saves weights to weights/ directory.

This is the first prototype training run for the project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ariannamethod'))
from chuck import ChuckOptimizer

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ── Architecture (tuned for ~20K params) ──────────────────────────────────────

D_MODEL = 24
N_HEADS = 4
N_LAYERS = 2
D_FF = 48
MAX_SEQ = 48
IMAGE_SIZE = 32       # smaller images for prototype
PATCH_SIZE = 8
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 4x4 = 16 patches
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE       # 192

# ── Training config ───────────────────────────────────────────────────────────

EPOCHS = 800
LR = 3e-3
WEIGHT_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(WEIGHT_DIR, exist_ok=True)


class SimpleTokenizer:
    """Character-level tokenizer"""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '?') for i in indices])


class VisionEncoder(nn.Module):
    """Minimal vision encoder for 64x64 images"""
    def __init__(self, d_model):
        super().__init__()
        self.patch_embedding = nn.Linear(PATCH_DIM, d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, N_PATCHES, d_model) * 0.02)

    def forward(self, images):
        B, C, H, W = images.shape
        # Patchify
        patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE) \
                        .unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.contiguous().view(B, C, -1, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous() \
                         .view(B, N_PATCHES, -1)
        x = self.patch_embedding(patches) + self.position_embedding
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention: text queries image patches"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, text_feat, image_feat):
        B, T, D = text_feat.shape
        _, I, _ = image_feat.shape
        Q = self.q(text_feat).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(image_feat).view(B, I, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(image_feat).view(B, I, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class VLMBlock(nn.Module):
    """Single VLM transformer block: self-attn + cross-attn + FFN"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = CrossModalAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, img, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        # Cross-attention
        cross_out = self.cross_attn(x, img)
        x = self.ln2(x + cross_out)
        # FFN
        ff_out = self.ffn(x)
        x = self.ln3(x + ff_out)
        return x


class MiniVLM(nn.Module):
    """~20K parameter VLM for prototype training"""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = D_MODEL
        self.max_seq = MAX_SEQ

        self.vision_encoder = VisionEncoder(D_MODEL)
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ, D_MODEL)
        self.blocks = nn.ModuleList([VLMBlock(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

    def forward(self, images, tokens):
        B, T = tokens.shape
        img_feat = self.vision_encoder(images)
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(tokens) + self.pos_emb(pos)

        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, img_feat, mask)

        x = self.ln_f(x)
        return self.head(x)

    def generate(self, image, tokenizer, max_len=30, temperature=0.8):
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        generated = [0]
        with torch.no_grad():
            for _ in range(max_len):
                tokens = torch.tensor([generated[-self.max_seq:]])
                logits = self.forward(image, tokens)
                probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
                generated.append(next_tok)
                if next_tok == 0:
                    break
        return tokenizer.decode(generated[1:])


def create_training_image():
    """Create a 32x32 synthetic image with red square in center"""
    img = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE) * 0.3
    c = IMAGE_SIZE // 2
    s = IMAGE_SIZE // 6
    img[0, c-s:c+s, c-s:c+s] = 0.8  # R
    img[1, c-s:c+s, c-s:c+s] = 0.15  # G
    img[2, c-s:c+s, c-s:c+s] = 0.15  # B
    return img


def main():
    print("=" * 60)
    print("  VLM Prototype Training")
    print("  Engine: torch (Python line) | Optimizer: Chuck")
    print("=" * 60)

    # Training data
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

    tokenizer = SimpleTokenizer(text)
    print(f"\n  Vocab size: {tokenizer.vocab_size}")
    print(f"  Text length: {len(text)} chars")

    model = MiniVLM(vocab_size=tokenizer.vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Chuck optimizer
    try:
        optimizer = ChuckOptimizer(model.parameters(), lr=LR)
        opt_name = "Chuck"
        print(f"  Optimizer: Chuck (self-aware, 9 levels)")
    except Exception as e:
        print(f"  Chuck init failed ({e}), falling back to Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        opt_name = "Adam"

    print(f"  Learning rate: {LR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Architecture: d={D_MODEL}, h={N_HEADS}, L={N_LAYERS}, ff={D_FF}")
    print(f"  Image: {IMAGE_SIZE}x{IMAGE_SIZE}, patches: {N_PATCHES}")

    # Encode training data
    input_ids = tokenizer.encode(text)
    image = create_training_image()

    # Training
    print(f"\n  Training...")
    print(f"  {'─' * 50}")

    model.train()
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Random window
        max_start = max(0, len(input_ids) - MAX_SEQ - 1)
        start = random.randint(0, max_start)
        end = start + min(MAX_SEQ, len(input_ids) - start - 1)
        if end <= start:
            continue

        x = torch.tensor([input_ids[start:end]])
        y = torch.tensor([input_ids[start+1:end+1]])
        img = image.unsqueeze(0)

        logits = model(img, x)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if opt_name == "Chuck":
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if epoch % 30 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - start_time
            print(f"   epoch {epoch:3d} | loss {loss_val:.4f} | "
                  f"best {best_loss:.4f} | {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"  {'─' * 50}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    print(f"  Best loss: {best_loss:.4f}")

    # Loss trend analysis
    if len(loss_history) >= 20:
        early = sum(loss_history[:20]) / 20
        late = sum(loss_history[-20:]) / 20
        print(f"  Loss trend: {early:.4f} (early) -> {late:.4f} (late)")
        print(f"  Improvement: {((early - late) / early * 100):.1f}%")

    # Save weights
    save_path = os.path.join(WEIGHT_DIR, 'vlm_prototype.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.chars,
        'vocab_size': tokenizer.vocab_size,
        'config': {
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'd_ff': D_FF,
            'max_seq': MAX_SEQ,
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
        },
        'training': {
            'epochs': EPOCHS,
            'lr': LR,
            'optimizer': opt_name,
            'final_loss': loss_history[-1],
            'best_loss': best_loss,
            'loss_history': loss_history,
        }
    }, save_path)
    print(f"\n  Weights saved to {save_path}")

    # Save training log
    log_path = os.path.join(WEIGHT_DIR, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'optimizer': opt_name,
            'epochs': EPOCHS,
            'lr': LR,
            'total_params': total_params,
            'final_loss': loss_history[-1],
            'best_loss': best_loss,
            'loss_history': loss_history,
            'elapsed_seconds': elapsed,
            'config': {
                'd_model': D_MODEL,
                'n_heads': N_HEADS,
                'n_layers': N_LAYERS,
                'd_ff': D_FF,
            }
        }, f, indent=2)
    print(f"  Training log saved to {log_path}")

    # Test generation
    print(f"\n  Testing generation:")
    print(f"  {'─' * 50}")

    for temp in [0.5, 0.8, 1.0]:
        caption = model.generate(image, tokenizer, max_len=40,
                                  temperature=temp)
        print(f"   temp={temp}: '{caption}'")

    print(f"\n  Done. Resonance is unbreakable.")


if __name__ == '__main__':
    main()
