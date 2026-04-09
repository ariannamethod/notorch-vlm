#!/usr/bin/env python3
"""
Simple VLM — Vision Language Model from scratch
Extended with Chuck Optimizer (Arianna Method)

Original fork: jiaquan301@163.com
notorch integration + Chuck optimizer: Arianna Method

Two lines of execution exist:
  Python line: torch tensors + Chuck optimizer (this file)
  C line:      notorch.c + chuck (see ariannamethod/train_vlm.c)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import os
import sys

# Chuck Optimizer — drop-in Adam replacement with 9 levels of self-awareness
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ariannamethod'))
from chuck import ChuckOptimizer

# Reproducibility
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
        """Encode text to token index list"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Decode token index list to text"""
        return ''.join([self.idx_to_char[i] for i in indices])


class SimpleVisionEncoder(nn.Module):
    """
    Simplified vision encoder — converts images to feature sequences.
    Core VLM component responsible for understanding image content.
    Similar to CLIP's image encoder, but minimal.
    """
    def __init__(self, image_size=224, patch_size=16, d_model=128, n_layers=2):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model

        self.n_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size  # RGB * patch_size^2

        print(f"  Vision encoder init:")
        print(f"   image size: {image_size}x{image_size}")
        print(f"   patch size: {patch_size}x{patch_size}")
        print(f"   patch count: {self.n_patches}")
        print(f"   output dim: {d_model}")

        # Patch embedding: image patches -> vectors
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)

        # Position embedding: spatial info for each patch
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        # Simplified transformer encoder
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def patchify(self, images):
        """
        Split image into patches.
        Input: (batch_size, 3, H, W)
        Output: (batch_size, n_patches, patch_dim)
        """
        batch_size, channels, height, width = images.shape
        assert height == width == self.image_size, \
            f"Image size should be {self.image_size}x{self.image_size}"

        patches = images.unfold(2, self.patch_size, self.patch_size) \
                        .unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1,
                                            self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(batch_size, self.n_patches, -1)
        return patches

    def forward(self, images):
        """
        Forward pass.
        Input: (batch_size, 3, H, W) image tensor
        Output: (batch_size, n_patches, d_model) visual features
        """
        patches = self.patchify(images)
        x = self.patch_embedding(patches)
        x = x + self.position_embedding
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores.masked_fill_(mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)

        return self.out_linear(attention_output)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention — the core VLM innovation.
    Allows text tokens to attend to image features,
    enabling vision-language interaction.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Query from text, Key/Value from image
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, text_features, image_features):
        """
        Cross-modal attention computation.
        text_features: (batch_size, text_len, d_model)
        image_features: (batch_size, image_len, d_model)
        """
        batch_size, text_len, d_model = text_features.shape
        _, image_len, _ = image_features.shape

        Q = self.q_linear(text_features)
        K = self.k_linear(image_features)
        V = self.v_linear(image_features)

        Q = Q.view(batch_size, text_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, image_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, image_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, text_len, d_model)

        return self.out_linear(attention_output)


class VLMTransformerBlock(nn.Module):
    """
    VLM-specific Transformer block.
    Adds cross-modal attention on top of standard Transformer.
    This is the key extension enabling image-text understanding.
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = CrossModalAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, text_features, image_features, causal_mask=None):
        # 1. Text self-attention
        attn_output = self.self_attention(text_features, causal_mask)
        text_features = self.norm1(text_features + attn_output)

        # 2. Cross-modal attention (text attends to image)
        cross_attn_output = self.cross_attention(text_features, image_features)
        text_features = self.norm2(text_features + cross_attn_output)

        # 3. Feed-forward network
        ff_output = self.feed_forward(text_features)
        text_features = self.norm3(text_features + ff_output)

        return text_features


class SimpleVLM(nn.Module):
    """
    Simplified Vision Language Model.
    Extends LLM with visual processing capability.
    Core idea: Image + Text -> Understanding -> Generation
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 max_seq_len=64, image_size=224, patch_size=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        print(f"  Initializing SimpleVLM:")
        print(f"   vocab size: {vocab_size}")
        print(f"   model dim: {d_model}")
        print(f"   attention heads: {n_heads}")
        print(f"   transformer layers: {n_layers}")

        # 1. Vision encoder: process images
        self.vision_encoder = SimpleVisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model
        )

        # 2. Text embedding (inherited from LLM)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # 3. VLM-specific Transformer blocks
        self.vlm_blocks = nn.ModuleList([
            VLMTransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])

        # 4. Output projection: generate text
        self.output_projection = nn.Linear(d_model, vocab_size)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"   total parameters: {total_params:,}")

    def forward(self, images, text_tokens):
        """
        Forward pass.
        images: (batch_size, 3, H, W) image tensor
        text_tokens: (batch_size, seq_len) text tokens
        """
        batch_size, seq_len = text_tokens.shape

        # 1. Extract visual features
        image_features = self.vision_encoder(images)

        # 2. Text embedding + positional encoding
        positions = torch.arange(seq_len, device=text_tokens.device) \
                         .unsqueeze(0).expand(batch_size, -1)
        text_features = self.token_embedding(text_tokens) + \
                        self.position_embedding(positions)

        # 3. Causal mask (prevent looking at future tokens)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # 4. VLM Transformer blocks: fuse vision and text
        for vlm_block in self.vlm_blocks:
            text_features = vlm_block(text_features, image_features, causal_mask)

        # 5. Output logits
        logits = self.output_projection(text_features)
        return logits

    def generate_caption(self, image, tokenizer, max_length=50, temperature=1.0):
        """
        Image caption generation.
        Core VLM application: look at image, generate description.
        """
        self.eval()

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        generated_tokens = [0]  # start token

        print(f"  Generating image caption...")

        with torch.no_grad():
            for step in range(max_length):
                current_tokens = torch.tensor([generated_tokens])
                if len(generated_tokens) >= self.max_seq_len:
                    current_tokens = torch.tensor(
                        [generated_tokens[-self.max_seq_len:]])

                logits = self.forward(image, current_tokens)
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated_tokens.append(next_token)

                if next_token == 0:
                    break

        try:
            caption = tokenizer.decode(generated_tokens[1:])
            print(f"  Done: '{caption}'")
            return caption
        except Exception:
            print("  Decode failed, returning raw tokens")
            return str(generated_tokens)


def create_dummy_image(size=(224, 224)):
    """Create a dummy image for demonstration."""
    image = np.random.rand(3, size[0], size[1]).astype(np.float32)

    # Add structured pattern — red square in center
    center_x, center_y = size[0] // 2, size[1] // 2
    for i in range(center_x - 20, center_x + 20):
        for j in range(center_y - 20, center_y + 20):
            if 0 <= i < size[0] and 0 <= j < size[1]:
                image[:, i, j] = [0.8, 0.2, 0.2]  # red square

    return torch.tensor(image)


def train_simple_vlm():
    """Train simple VLM with Chuck optimizer."""
    print("  Starting VLM training")
    print("=" * 60)

    # Training data
    text = """
    This is a red square. There is a red object in the center of the image.
    The red square is located in the center. This object is red.
    The image shows a red square. The center area is red.
    """

    print(f"  Training text: {text[:50]}...")

    tokenizer = SimpleTokenizer(text)
    model = SimpleVLM(vocab_size=tokenizer.vocab_size, d_model=64,
                      n_heads=4, n_layers=2)

    input_ids = tokenizer.encode(text)
    dummy_image = create_dummy_image()

    # Chuck optimizer (Adam as fallback)
    try:
        optimizer = ChuckOptimizer(model.parameters(), lr=0.001)
        optimizer_name = "Chuck"
        print("  Optimizer: Chuck (self-aware, 9 levels)")
    except Exception:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer_name = "Adam (fallback)"
        print("  Optimizer: Adam (fallback)")

    model.train()
    print(f"\n  Training with {optimizer_name}...")

    for epoch in range(50):
        start_idx = random.randint(
            0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + min(model.max_seq_len,
                                  len(input_ids) - start_idx - 1)

        x_text = torch.tensor([input_ids[start_idx:end_idx]])
        y_text = torch.tensor([input_ids[start_idx + 1:end_idx + 1]])
        x_image = dummy_image.unsqueeze(0)

        logits = model(x_image, x_text)
        loss = F.cross_entropy(
            logits.view(-1, tokenizer.vocab_size), y_text.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if optimizer_name == "Chuck":
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}, Loss: {loss.item():.4f}")

    print("  Training complete!")

    # Test caption generation
    print(f"\n  Testing image caption generation:")
    print("-" * 40)

    test_image = create_dummy_image()
    caption = model.generate_caption(
        test_image, tokenizer, max_length=20, temperature=0.8)

    print(f"  Input: dummy red square image")
    print(f"  Generated caption: {caption}")

    return model, tokenizer


def demonstrate_vlm_components():
    """Demonstrate VLM component internals."""
    print("\n" + "=" * 60)
    print("  VLM Component Demo")
    print("=" * 60)

    # 1. Vision encoder demo
    print("\n  1. Vision encoder:")
    vision_encoder = SimpleVisionEncoder(image_size=224, patch_size=16,
                                          d_model=64)

    dummy_image = create_dummy_image().unsqueeze(0)
    visual_features = vision_encoder(dummy_image)

    print(f"   Input image shape: {dummy_image.shape}")
    print(f"   Output feature shape: {visual_features.shape}")
    print(f"   Meaning: {visual_features.shape[1]} image patches, "
          f"each represented as {visual_features.shape[2]}-dim vector")

    # 2. Cross-modal attention demo
    print("\n  2. Cross-modal attention:")
    cross_attention = CrossModalAttention(d_model=64, n_heads=4)

    text_features = torch.randn(1, 10, 64)  # 10 text tokens
    attended_features = cross_attention(text_features, visual_features)

    print(f"   Text feature shape: {text_features.shape}")
    print(f"   Image feature shape: {visual_features.shape}")
    print(f"   Attention output shape: {attended_features.shape}")
    print(f"   Text tokens attend to image patches to get visual info")


if __name__ == "__main__":
    print("  Simple VLM — powered by notorch core + Chuck optimizer")
    print("=" * 60)
    print("Vision Language Model from scratch")
    print("Core: Image understanding + Text generation = See and describe")
    print()

    demonstrate_vlm_components()
    model, tokenizer = train_simple_vlm()

    print("\n" + "=" * 60)
    print("  VLM demo complete!")
    print("=" * 60)
    print("What you learned:")
    print("  How to extend LLM to VLM")
    print("  Vision encoder internals")
    print("  Cross-modal attention mechanism")
    print("  Image caption generation")
    print("\n  Core VLM idea:")
    print("   Vision encoder extracts visual features")
    print("   Cross-modal attention fuses vision and text")
    print("   Language model generates description text")
