#!/usr/bin/env python3
"""
Minimal VLM — core implementation
Vision Language Model extending LLM, highlighting core logic.
Powered by Chuck Optimizer (Arianna Method).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ariannamethod'))
from chuck import ChuckOptimizer

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


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
        return ''.join([self.idx_to_char[i] for i in indices])


class VisionEncoder(nn.Module):
    """Vision encoder: image -> feature sequence"""
    def __init__(self, image_size=224, patch_size=16, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embedding = nn.Linear(patch_dim, d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=4, dim_feedforward=d_model * 4,
                batch_first=True),
            num_layers=2
        )

    def patchify(self, images):
        """Split image into patches"""
        B, C, H, W = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size) \
                        .unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(
            B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous() \
                         .view(B, self.n_patches, -1)
        return patches

    def forward(self, images):
        patches = self.patchify(images)
        x = self.patch_embedding(patches) + self.position_embedding
        return self.transformer(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention: text attends to image"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, text_features, image_features):
        B, T, D = text_features.shape
        _, I, _ = image_features.shape

        Q = self.q_linear(text_features).view(
            B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(image_features).view(
            B, I, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(image_features).view(
            B, I, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_linear(attn_output)


class VLMBlock(nn.Module):
    """VLM Transformer block"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.cross_attn = CrossModalAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, text_features, image_features, causal_mask=None):
        # Text self-attention
        attn_out, _ = self.self_attn(
            text_features, text_features, text_features,
            attn_mask=causal_mask)
        text_features = self.norm1(text_features + attn_out)

        # Cross-modal attention
        cross_out = self.cross_attn(text_features, image_features)
        text_features = self.norm2(text_features + cross_out)

        # Feed-forward network
        ffn_out = self.ffn(text_features)
        text_features = self.norm3(text_features + ffn_out)

        return text_features


class SimpleVLM(nn.Module):
    """Simplified Vision Language Model"""
    def __init__(self, vocab_size, d_model=128, n_layers=2, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.vision_encoder = VisionEncoder(d_model=d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.vlm_blocks = nn.ModuleList(
            [VLMBlock(d_model) for _ in range(n_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, images, text_tokens):
        B, T = text_tokens.shape

        # Process image and text
        image_features = self.vision_encoder(images)
        positions = torch.arange(T, device=text_tokens.device) \
                         .unsqueeze(0).expand(B, -1)
        text_features = self.token_embedding(text_tokens) + \
                        self.position_embedding(positions)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()

        # VLM processing
        for block in self.vlm_blocks:
            text_features = block(text_features, image_features, causal_mask)

        return self.output_projection(text_features)

    def generate(self, image, tokenizer, max_length=20):
        """Image caption generation"""
        self.eval()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        generated = [0]  # start token

        with torch.no_grad():
            for _ in range(max_length):
                tokens = torch.tensor([generated[-self.max_seq_len:]])
                logits = self.forward(image, tokens)
                next_token = torch.multinomial(
                    F.softmax(logits[0, -1, :], dim=-1), 1).item()
                generated.append(next_token)
                if next_token == 0:  # end token
                    break

        return tokenizer.decode(generated[1:])


def create_demo_image():
    """Create demo image with red square in center"""
    image = np.random.rand(3, 224, 224).astype(np.float32)
    center = 112
    image[0, center - 20:center + 20, center - 20:center + 20] = 0.8  # R
    image[1, center - 20:center + 20, center - 20:center + 20] = 0.2  # G
    image[2, center - 20:center + 20, center - 20:center + 20] = 0.2  # B
    return torch.tensor(image)


def main():
    print("  Minimal VLM — core implementation")
    print("=" * 50)

    # Data and model
    text = "This is a red square. There is a red object in the center. " \
           "The red square is in the middle."
    tokenizer = SimpleTokenizer(text)
    model = SimpleVLM(vocab_size=tokenizer.vocab_size, d_model=64)

    print(f"  Model info:")
    print(f"   vocab size: {tokenizer.vocab_size}")
    print(f"   parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    input_ids = tokenizer.encode(text)
    image = create_demo_image()

    # Chuck optimizer (Adam as fallback)
    try:
        optimizer = ChuckOptimizer(model.parameters(), lr=0.001)
        opt_name = "Chuck"
        print(f"\n  Optimizer: Chuck (self-aware)")
    except Exception:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        opt_name = "Adam (fallback)"
        print(f"\n  Optimizer: Adam (fallback)")

    print(f"  Training...")
    model.train()
    for epoch in range(30):
        start = random.randint(
            0, max(0, len(input_ids) - model.max_seq_len - 1))
        end = start + min(model.max_seq_len, len(input_ids) - start - 1)

        x_text = torch.tensor([input_ids[start:end]])
        y_text = torch.tensor([input_ids[start + 1:end + 1]])
        x_image = image.unsqueeze(0)

        logits = model(x_image, x_text)
        loss = F.cross_entropy(
            logits.view(-1, tokenizer.vocab_size), y_text.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if opt_name == "Chuck":
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}: Loss = {loss.item():.3f}")

    # Test generation
    print(f"\n  Image caption generation:")
    caption = model.generate(image, tokenizer)
    print(f"   Result: '{caption}'")

    print(f"\n  Demo complete!")
    print("  Core concepts:")
    print("   - Vision encoder: image -> patch feature sequence")
    print("   - Cross-modal attention: text attends to image features")
    print("   - VLM block: fuses vision and language information")
    print("   - Generation: produces description based on image content")


if __name__ == "__main__":
    main()
