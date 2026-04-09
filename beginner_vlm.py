#!/usr/bin/env python3
"""
Beginner VLM Tutorial — companion demo code
============================================

This file provides a complete VLM implementation with detailed explanations.
Powered by Chuck Optimizer (Arianna Method).

Running this file shows you:
1. Complete VLM implementation with educational comments
2. Detailed explanation of each component
3. Training process visualization and analysis
4. Deep understanding of model behavior
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

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

print("  Beginner VLM Tutorial")
print("=" * 60)
print("Educational VLM demo with Chuck optimizer")
print("=" * 60)

# ============================================================================
# Part 1: Core Components
# ============================================================================


class SimpleTokenizer:
    """
    Simple character-level tokenizer.

    Purpose: convert text to numbers so the computer can understand.
    Example: 'red square' -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    def __init__(self, text):
        print("  Initializing tokenizer...")

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"   Vocab size: {self.vocab_size}")
        sample = self.chars[:10]
        print(f"   Vocab sample: {sample}..."
              if len(self.chars) > 10 else f"   Vocab: {self.chars}")

    def encode(self, text):
        """Encode text to token index list"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Decode token index list to text"""
        return ''.join([self.idx_to_char[i] for i in indices])


class SimpleVisionEncoder(nn.Module):
    """
    Simplified vision encoder — the VLM's "eyes".

    Purpose: convert images to feature sequences so the model can "see".
    How it works:
    1. Split image into patches (like puzzle pieces)
    2. Convert each patch into a feature vector
    3. Add position information
    4. Process feature relationships with Transformer

    Input:  image (3, 224, 224)
    Output: feature sequence (196, d_model)
    """
    def __init__(self, image_size=224, patch_size=16, d_model=128, n_layers=2):
        super().__init__()
        print(f"  Initializing vision encoder...")

        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model

        # How many patches the image will be split into
        self.num_patches = (image_size // patch_size) ** 2  # 14x14 = 196
        self.patch_dim = patch_size * patch_size * 3  # 16x16x3 = 768

        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Patch size: {patch_size}x{patch_size}")
        print(f"   Total patches: {self.num_patches}")
        print(f"   Pixels per patch: {self.patch_dim}")

        # Linear layer: convert image patches to feature vectors
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)

        # Position encoding: tell the model where each patch is in the image
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model))

        # Transformer encoder: let patches "communicate"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        print(f"   Feature dim: {d_model}")
        print(f"   Transformer layers: {n_layers}")

    def forward(self, x):
        """
        Complete image processing pipeline.

        Steps:
        1. Split image:    (B, 3, 224, 224) -> (B, 196, 768)
        2. Feature mapping: (B, 196, 768) -> (B, 196, 128)
        3. Position encoding: add spatial information
        4. Transformer: understand inter-patch relationships
        """
        B = x.shape[0]

        # Step 1: split image into patches
        x = x.unfold(2, self.patch_size, self.patch_size) \
             .unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, 3, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.num_patches, -1)  # (B, 196, 768)

        # Step 2: convert each patch to feature vector
        x = self.patch_embedding(x)  # (B, 196, 128)

        # Step 3: add position encoding
        x = x + self.position_embedding

        # Step 4: Transformer processing
        x = self.transformer(x)  # (B, 196, 128)

        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention — the bridge between text and image.

    Purpose: build associations between text and image.
    How it works:
    - Text acts as "query": what information do I need?
    - Image acts as "key/value": what information can I provide?
    - Select the most relevant image regions based on relevance.

    Example:
    - Generating "red" -> focuses on red regions in image
    - Generating "square" -> focuses on shape boundaries in image
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        print(f"  Initializing cross-modal attention...")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Multi-head attention: understand image-text relations
        # from multiple angles simultaneously
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)

        print(f"   Feature dim: {d_model}")
        print(f"   Attention heads: {n_heads}")
        print(f"   Per-head dim: {self.head_dim}")

    def forward(self, text_features, image_features):
        """
        Cross-modal attention computation.

        Input:
        - text_features: (B, seq_len, d_model)
        - image_features: (B, num_patches, d_model)

        Output:
        - attended_features: (B, seq_len, d_model)
        """
        # Cross-modal attention: text attends to relevant image regions
        attended_features, attention_weights = self.multihead_attn(
            query=text_features,     # text asks: what info do I need?
            key=image_features,      # image answers: here's what I have
            value=image_features     # image content
        )

        # Residual connection: preserve original text information
        attended_features = self.norm(attended_features + text_features)

        return attended_features, attention_weights


class VLMTransformerBlock(nn.Module):
    """
    VLM Transformer block — core understanding and fusion unit.

    Purpose: deep understanding and fusion of text and image information.
    Pipeline:
    1. Self-attention: understand intra-text relationships
    2. Cross-modal attention: text attends to relevant image regions
    3. Feed-forward network: further process fused information

    Each block deepens the model's understanding.
    Multiple stacked blocks form deep comprehension capability.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        print(f"  Initializing VLM Transformer block...")

        # Self-attention: understand relationships within text sequence
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        # Cross-modal attention: text attends to relevant image regions
        self.cross_attention = CrossModalAttention(d_model, n_heads)

        # Feed-forward network: further process fused information
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # expand dimension
            nn.ReLU(),                         # nonlinear activation
            nn.Linear(d_model * 4, d_model)    # compress back
        )

        # Layer normalization: stabilize training, accelerate convergence
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        print(f"   Feature dim: {d_model}")
        print(f"   Attention heads: {n_heads}")

    def forward(self, text_features, image_features):
        """
        VLM block full processing pipeline.

        Three steps:
        1. Self-attention + residual connection
        2. Cross-modal attention + residual connection
        3. Feed-forward network + residual connection
        """
        # Step 1: self-attention — understand intra-text relationships
        attn_output, _ = self.self_attention(
            text_features, text_features, text_features)
        text_features = self.norm1(text_features + attn_output)

        # Step 2: cross-modal attention — text attends to image
        cross_attn_output, attention_weights = self.cross_attention(
            text_features, image_features)
        text_features = self.norm2(text_features + cross_attn_output)

        # Step 3: feed-forward network — further processing
        ff_output = self.feed_forward(text_features)
        text_features = self.norm3(text_features + ff_output)

        return text_features, attention_weights


class SimpleVLM(nn.Module):
    """
    Simple Vision Language Model — integrating all components.

    An AI model that can look at images and describe them.

    Architecture:
    1. Vision encoder: processes images
    2. Text embedding: processes text
    3. VLM blocks: deep fusion and understanding
    4. Output layer: generates text

    Capabilities:
    - Image captioning: look at image, describe content
    - Visual QA: answer questions based on image
    - Multimodal understanding: understand images and text together
    """
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=6,
                 max_seq_len=100):
        super().__init__()
        print(f"  Initializing SimpleVLM...")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        print("   Initializing components...")

        # 1. Vision encoder: the VLM's "eyes"
        self.vision_encoder = SimpleVisionEncoder(d_model=d_model)

        # 2. Text embedding: convert text to feature vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model))

        # 3. VLM processing blocks: core understanding and fusion
        self.vlm_blocks = nn.ModuleList([
            VLMTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # 4. Output layer: convert features back to text
        self.output_projection = nn.Linear(d_model, vocab_size)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Vocab size: {vocab_size}")
        print(f"   Model dim: {d_model}")
        print(f"   Attention heads: {n_heads}")
        print(f"   VLM layers: {n_layers}")
        print(f"   Total parameters: {total_params:,}")

    def forward(self, image, text_tokens):
        """
        Full forward pass.

        Pipeline:
        1. Image -> visual features
        2. Text -> text features
        3. Multi-layer VLM processing -> deep fusion
        4. Output layer -> predict next word
        """
        B, seq_len = text_tokens.shape

        # Step 1: process image, extract visual features
        image_features = self.vision_encoder(image)

        # Step 2: process text, convert to feature vectors
        text_features = self.token_embedding(text_tokens)

        # Add position encoding
        text_features = text_features + \
            self.position_embedding[:, :seq_len, :]

        # Step 3: multi-layer VLM processing
        attention_maps = []
        for vlm_block in self.vlm_blocks:
            text_features, attention_weights = vlm_block(
                text_features, image_features)
            attention_maps.append(attention_weights)

        # Step 4: output layer, predict next word probability distribution
        logits = self.output_projection(text_features)

        return logits, attention_maps

    def generate(self, image, tokenizer, prompt="", max_length=20,
                 temperature=1.0):
        """
        Generate image description.

        Strategy:
        1. Start from prompt (if any)
        2. Predict next word one at a time
        3. Stop when max length reached or end token generated

        Temperature:
        - Low (0.5): more certain, conservative text
        - High (1.5): more random, creative text
        """
        self.eval()

        with torch.no_grad():
            if prompt:
                generated = tokenizer.encode(prompt)
            else:
                generated = []

            for _ in range(max_length):
                if len(generated) == 0:
                    current_tokens = torch.tensor([[0]], dtype=torch.long)
                else:
                    current_tokens = torch.tensor(
                        [generated[-self.max_seq_len:]], dtype=torch.long)

                logits, _ = self.forward(image, current_tokens)
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)

                if len(generated) > 1 and next_token == generated[0]:
                    break

        return tokenizer.decode(generated)


# ============================================================================
# Part 2: Demo and Analysis Functions
# ============================================================================


def create_demo_image():
    """
    Create a demo red square image.
    Content: 224x224 image with red square in center.
    Format: RGB 3-channel, value range 0-1.
    """
    print("  Creating demo image...")

    image = torch.ones(3, 224, 224) * 0.5  # gray background

    # Draw red square in center
    center = 112
    size = 40
    image[0, center - size:center + size,
          center - size:center + size] = 0.8  # R
    image[1, center - size:center + size,
          center - size:center + size] = 0.1  # G
    image[2, center - size:center + size,
          center - size:center + size] = 0.1  # B

    print("    Created 224x224 image with red square in center")
    return image.unsqueeze(0)  # add batch dimension


def analyze_attention_patterns(attention_maps, tokenizer, text_tokens,
                                step_name=""):
    """
    Analyze attention patterns — what is the model "looking at"?

    Analysis:
    - Which image regions each word attends to
    - Concentration level of attention
    - Attention changes across layers
    """
    print(f"\n  {step_name}Attention pattern analysis:")

    if len(attention_maps) > 0:
        last_attention = attention_maps[-1][0]

        for i, token_id in enumerate(
                text_tokens[0][:min(5, len(text_tokens[0]))]):
            if i < last_attention.shape[0]:
                word = tokenizer.decode([token_id.item()])
                attention_weights = last_attention[i]

                max_patch = torch.argmax(attention_weights).item()
                max_val = torch.max(attention_weights).item()

                probs = F.softmax(attention_weights, dim=0)
                entropy = -torch.sum(
                    probs * torch.log(probs + 1e-10)).item()

                print(f"   '{word}': top patch {max_patch} "
                      f"(weight:{max_val:.3f}, entropy:{entropy:.2f})")


def train_vlm_step_by_step():
    """
    Train VLM step by step, showing the complete learning process.

    Training phases:
    1. Data preparation
    2. Model initialization
    3. Training loop
    4. Generation test
    5. Attention analysis
    """
    print("\n  Starting VLM training demo")
    print("=" * 50)

    # Step 1: prepare training data
    print("\n  Step 1: Preparing training data")
    training_text = ("This is a red square. There is a red object in the "
                     "center of the image. The red square is located in the "
                     "center. This object is red. The square is red. "
                     "The red square is in the image.")

    tokenizer = SimpleTokenizer(training_text)
    text_tokens = torch.tensor(
        [tokenizer.encode(training_text)], dtype=torch.long)

    print(f"   Training text: '{training_text[:40]}...'")
    print(f"   Text length: {len(training_text)} characters")
    print(f"   Token count: {text_tokens.shape[1]}")

    # Step 2: create model
    print("\n  Step 2: Creating VLM model")
    model = SimpleVLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=4
    )

    # Step 3: create image and optimizer
    print("\n  Step 3: Preparing training environment")
    image = create_demo_image()

    # Chuck optimizer (Adam as fallback)
    try:
        optimizer = ChuckOptimizer(model.parameters(), lr=0.001)
        opt_name = "Chuck"
        print("   Optimizer: Chuck (self-aware, 9 levels)")
    except Exception:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        opt_name = "Adam (fallback)"
        print("   Optimizer: Adam (fallback)")

    criterion = nn.CrossEntropyLoss()

    # Step 4: training loop
    print("\n  Step 4: Training")
    print("   Progress:")

    model.train()
    for epoch in range(41):
        optimizer.zero_grad()

        input_tokens = text_tokens[:, :-1]
        target_tokens = text_tokens[:, 1:]

        logits, attention_maps = model(image, input_tokens)
        loss = criterion(
            logits.reshape(-1, tokenizer.vocab_size),
            target_tokens.reshape(-1))

        loss.backward()
        if opt_name == "Chuck":
            optimizer.step(loss=loss.item())
        else:
            optimizer.step()

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}: Loss = {loss.item():.4f}")

            if epoch in [0, 20, 40]:
                analyze_attention_patterns(
                    attention_maps, tokenizer, input_tokens,
                    f"Epoch {epoch} "
                )

    # Step 5: test generation
    print("\n  Step 5: Testing text generation")
    model.eval()

    for temp in [0.5, 1.0, 1.5]:
        generated_text = model.generate(
            image, tokenizer, max_length=10, temperature=temp)
        print(f"   Temperature {temp}: '{generated_text}'")

    print("\n  Training demo complete!")
    return model, tokenizer, image


def demonstrate_components():
    """
    Demonstrate how each VLM component works.

    Demos:
    1. Tokenizer encode/decode
    2. Vision encoder image processing
    3. Cross-modal attention computation
    4. Complete model forward pass
    """
    print("\n  VLM Component Demo")
    print("=" * 40)

    # Demo 1: Tokenizer
    print("\n  Demo 1: Tokenizer")
    text = "red square"
    tokenizer = SimpleTokenizer("red square in the image")
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"   Original: '{text}'")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")

    # Demo 2: Vision encoder
    print("\n  Demo 2: Vision encoder")
    vision_encoder = SimpleVisionEncoder(d_model=128)
    image = create_demo_image()

    print(f"   Input image shape: {image.shape}")

    with torch.no_grad():
        image_features = vision_encoder(image)
        print(f"   Output feature shape: {image_features.shape}")
        compression = (3 * 224 * 224) / (
            image_features.shape[1] * image_features.shape[2])
        print(f"   Compression ratio: {compression:.1f}:1")

    # Demo 3: Cross-modal attention
    print("\n  Demo 3: Cross-modal attention")
    cross_attention = CrossModalAttention(d_model=128, n_heads=8)

    text_features = torch.randn(1, 5, 128)  # 5 text tokens

    with torch.no_grad():
        attended_features, attention_weights = cross_attention(
            text_features, image_features)

        print(f"   Text feature shape: {text_features.shape}")
        print(f"   Image feature shape: {image_features.shape}")
        print(f"   Attention weight shape: {attention_weights.shape}")
        print(f"   Fused feature shape: {attended_features.shape}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main program — run full VLM demo."""
    print("\n  Beginner VLM tutorial companion demo")
    print("Now you can:")
    print("  Understand how real VLMs work")
    print("  See implementation consistent with the technical blog")
    print("  Experience the complete training and generation process")
    print("  Analyze the model's attention mechanism")

    model, tokenizer, image = train_vlm_step_by_step()

    print("\n  Learning summary:")
    print("What you learned:")
    print("  - Complete VLM architecture and implementation")
    print("  - Specific role of each component")
    print("  - Detailed training process analysis")
    print("  - How attention mechanisms work")
    print("  - How to generate image descriptions")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("  Note: no GPU detected, running on CPU (slower)")

    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--interactive":
        print("  Interactive mode not available in this version.")
        print("  Running full demo instead.")
    main()
