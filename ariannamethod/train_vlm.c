// train_vlm.c — Minimal VLM training in pure C using notorch + Chuck
// Part of simple_vlm project (Arianna Method)
//
// C line: notorch core + Chuck optimizer
// ~20K parameters, character-level, synthetic image data
//
// Build:
//   cc -std=c11 -O2 -I. -o train_vlm train_vlm.c notorch.c -lm
//
// Run:
//   ./train_vlm

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ── Architecture config ──────────────────────────────────────────────────────

#define IMAGE_DIM    16      // flattened synthetic "image" feature dimension
#define D_MODEL      32      // model hidden dimension
#define N_HEADS       4      // attention heads
#define HEAD_DIM     (D_MODEL / N_HEADS)
#define D_FF         64      // feed-forward intermediate dimension
#define MAX_SEQ      32      // max sequence length
#define VOCAB_SIZE   64      // character vocabulary (ASCII printable subset)
#define N_LAYERS      2      // transformer layers
#define N_PATCHES     4      // synthetic "image patches"

// ── Training config ─────────────────────────────────────────────────────────

#define LR           0.003f
#define EPOCHS       200
#define SAVE_PATH    "../weights/vlm_notorch.bin"

// ── Model weights (global indices into tape) ─────────────────────────────────

typedef struct {
    // Vision projection: IMAGE_DIM -> D_MODEL
    int vis_proj_w;    // [D_MODEL, IMAGE_DIM]
    int vis_proj_b;    // [D_MODEL]
    int vis_pos;       // [N_PATCHES * D_MODEL]

    // Text embedding
    int wte;           // [VOCAB_SIZE, D_MODEL]
    int wpe;           // [MAX_SEQ, D_MODEL]

    // Per-layer weights
    struct {
        // Self-attention Q/K/V + output
        int wq, wk, wv, wo;  // each [D_MODEL, D_MODEL]
        int ln1_g, ln1_b;    // LayerNorm

        // Cross-attention Q/K/V + output
        int cq, ck, cv, co;  // each [D_MODEL, D_MODEL]
        int ln2_g, ln2_b;    // LayerNorm

        // FFN
        int ff1, ff2;        // [D_MODEL, D_FF] and [D_FF, D_MODEL]
        int ln3_g, ln3_b;    // LayerNorm
    } layers[N_LAYERS];

    // Output projection (weight-tied with wte)
    int ln_final_g, ln_final_b;

    int total_params;
} VLMWeights;

// ── Simple random helper (stdlib-based) ──────────────────────────────────────

static float vlm_rand_uniform(void) {
    return (float)rand() / (float)RAND_MAX;
}

// ── Simple char tokenizer ────────────────────────────────────────────────────

static char vocab[VOCAB_SIZE];
static int  vocab_len = 0;
static int  char_to_id[256];

static void init_vocab(const char* text) {
    memset(char_to_id, -1, sizeof(char_to_id));
    int used[256] = {0};
    for (int i = 0; text[i]; i++) used[(unsigned char)text[i]] = 1;
    vocab_len = 0;
    for (int c = 0; c < 256 && vocab_len < VOCAB_SIZE; c++) {
        if (used[c]) {
            vocab[vocab_len] = (char)c;
            char_to_id[c] = vocab_len;
            vocab_len++;
        }
    }
    printf("  Vocab: %d characters\n", vocab_len);
}

static int encode_char(char c) {
    int id = char_to_id[(unsigned char)c];
    return id >= 0 ? id : 0;
}

// ── Create synthetic image features ──────────────────────────────────────────

static void create_synthetic_image(float* out, int len) {
    // Red square pattern encoded as feature vector
    for (int i = 0; i < len; i++) {
        out[i] = 0.1f;
    }
    // "Red" signal in first quarter
    for (int i = 0; i < len / 4; i++) {
        out[i] = 0.8f;
    }
    // "Square" signal in second quarter
    for (int i = len / 4; i < len / 2; i++) {
        out[i] = 0.6f;
    }
}

// ── Initialize model ─────────────────────────────────────────────────────────

static VLMWeights init_model(void) {
    VLMWeights w;
    int total = 0;

    nt_tape_start();

    // Vision projection
    nt_tensor* vw = nt_tensor_new2d(D_MODEL, IMAGE_DIM);
    nt_tensor_xavier(vw, IMAGE_DIM, D_MODEL);
    w.vis_proj_w = nt_tape_param(vw);
    total += vw->len;

    nt_tensor* vb = nt_tensor_new(D_MODEL);
    nt_tensor_fill(vb, 0.0f);
    w.vis_proj_b = nt_tape_param(vb);
    total += vb->len;

    nt_tensor* vp = nt_tensor_new(N_PATCHES * D_MODEL);
    nt_tensor_rand(vp, 0.02f);
    w.vis_pos = nt_tape_param(vp);
    nt_tape_no_decay(w.vis_pos);
    total += vp->len;

    // Text embeddings
    nt_tensor* wte = nt_tensor_new2d(VOCAB_SIZE, D_MODEL);
    nt_tensor_rand(wte, 0.02f);
    w.wte = nt_tape_param(wte);
    nt_tape_no_decay(w.wte);
    total += wte->len;

    nt_tensor* wpe = nt_tensor_new2d(MAX_SEQ, D_MODEL);
    nt_tensor_rand(wpe, 0.02f);
    w.wpe = nt_tape_param(wpe);
    nt_tape_no_decay(w.wpe);
    total += wpe->len;

    // Transformer layers
    for (int l = 0; l < N_LAYERS; l++) {
        // Helper: allocate a weight matrix and register as param
        #define ALLOC_MAT(field, rows, cols) do { \
            nt_tensor* _t = nt_tensor_new2d(rows, cols); \
            nt_tensor_xavier(_t, cols, rows); \
            w.layers[l].field = nt_tape_param(_t); \
            total += _t->len; \
        } while(0)

        // Helper: allocate a bias/norm vector and register as no-decay param
        #define ALLOC_VEC(field, size, val) do { \
            nt_tensor* _t = nt_tensor_new(size); \
            nt_tensor_fill(_t, val); \
            w.layers[l].field = nt_tape_param(_t); \
            nt_tape_no_decay(w.layers[l].field); \
            total += _t->len; \
        } while(0)

        // Self-attention
        ALLOC_MAT(wq, D_MODEL, D_MODEL);
        ALLOC_MAT(wk, D_MODEL, D_MODEL);
        ALLOC_MAT(wv, D_MODEL, D_MODEL);
        ALLOC_MAT(wo, D_MODEL, D_MODEL);
        ALLOC_VEC(ln1_g, D_MODEL, 1.0f);
        ALLOC_VEC(ln1_b, D_MODEL, 0.0f);

        // Cross-attention
        ALLOC_MAT(cq, D_MODEL, D_MODEL);
        ALLOC_MAT(ck, D_MODEL, D_MODEL);
        ALLOC_MAT(cv, D_MODEL, D_MODEL);
        ALLOC_MAT(co, D_MODEL, D_MODEL);
        ALLOC_VEC(ln2_g, D_MODEL, 1.0f);
        ALLOC_VEC(ln2_b, D_MODEL, 0.0f);

        // FFN
        ALLOC_MAT(ff1, D_FF, D_MODEL);
        ALLOC_MAT(ff2, D_MODEL, D_FF);
        ALLOC_VEC(ln3_g, D_MODEL, 1.0f);
        ALLOC_VEC(ln3_b, D_MODEL, 0.0f);

        #undef ALLOC_MAT
        #undef ALLOC_VEC
    }

    // Final layer norm
    nt_tensor* lng = nt_tensor_new(D_MODEL);
    nt_tensor_fill(lng, 1.0f);
    w.ln_final_g = nt_tape_param(lng);
    nt_tape_no_decay(w.ln_final_g);
    total += lng->len;

    nt_tensor* lnb = nt_tensor_new(D_MODEL);
    nt_tensor_fill(lnb, 0.0f);
    w.ln_final_b = nt_tape_param(lnb);
    nt_tape_no_decay(w.ln_final_b);
    total += lnb->len;

    w.total_params = total;
    printf("  Model initialized: %d parameters\n", total);
    return w;
}

// ── Forward pass ─────────────────────────────────────────────────────────────

static int forward_pass(VLMWeights* w, int* tokens, int T,
                        float* image_features) {
    // 1. Vision projection: [N_PATCHES, IMAGE_DIM] -> [N_PATCHES, D_MODEL]
    //    (simplified: process each patch through linear layer)
    int vis_idx = -1;
    {
        // Create image input tensor
        nt_tensor* img = nt_tensor_new(N_PATCHES * IMAGE_DIM);
        for (int p = 0; p < N_PATCHES; p++) {
            for (int d = 0; d < IMAGE_DIM; d++) {
                img->data[p * IMAGE_DIM + d] =
                    image_features[p * IMAGE_DIM + d];
            }
        }
        int img_idx = nt_tape_get()->count;
        nt_tape_get()->entries[img_idx].output = img;
        nt_tape_get()->entries[img_idx].op = NT_OP_NONE;
        nt_tape_get()->count++;

        // Project each patch
        vis_idx = nt_seq_linear(w->vis_proj_w, img_idx, N_PATCHES);

        // Add position embedding
        vis_idx = nt_add(vis_idx, w->vis_pos);
    }

    // 2. Text embedding
    nt_tensor* tok_t = nt_tensor_new(T);
    for (int i = 0; i < T; i++) tok_t->data[i] = (float)tokens[i];
    int tok_idx = nt_tape_get()->count;
    nt_tape_get()->entries[tok_idx].output = tok_t;
    nt_tape_get()->entries[tok_idx].op = NT_OP_NONE;
    nt_tape_get()->count++;

    int x = nt_seq_embedding(w->wte, w->wpe, tok_idx, T, D_MODEL);

    // 3. Transformer layers
    for (int l = 0; l < N_LAYERS; l++) {
        // Self-attention
        int q = nt_seq_linear(w->layers[l].wq, x, T);
        int k = nt_seq_linear(w->layers[l].wk, x, T);
        int v = nt_seq_linear(w->layers[l].wv, x, T);
        int attn = nt_mh_causal_attention(q, k, v, T, HEAD_DIM);
        attn = nt_seq_linear(w->layers[l].wo, attn, T);
        x = nt_add(x, attn);
        x = nt_seq_layernorm(x, w->layers[l].ln1_g,
                              w->layers[l].ln1_b, T, D_MODEL);

        // Cross-attention (text attends to image)
        int cq = nt_seq_linear(w->layers[l].cq, x, T);
        int ck = nt_seq_linear(w->layers[l].ck, vis_idx, N_PATCHES);
        int cv = nt_seq_linear(w->layers[l].cv, vis_idx, N_PATCHES);

        // Simple cross-attention: Q from text, K/V from image
        // Use scaled dot-product (simplified — no causal mask for cross-attn)
        {
            nt_tensor* q_t = nt_tape_get()->entries[cq].output;
            nt_tensor* k_t = nt_tape_get()->entries[ck].output;
            nt_tensor* v_t = nt_tape_get()->entries[cv].output;

            nt_tensor* out = nt_tensor_new(T * D_MODEL);
            float scale = 1.0f / sqrtf((float)HEAD_DIM);

            for (int h = 0; h < N_HEADS; h++) {
                for (int t = 0; t < T; t++) {
                    // Compute attention scores for this head
                    float scores[N_PATCHES];
                    float max_s = -1e30f;
                    for (int p = 0; p < N_PATCHES; p++) {
                        float dot = 0.0f;
                        for (int d = 0; d < HEAD_DIM; d++) {
                            int qi = t * D_MODEL + h * HEAD_DIM + d;
                            int ki = p * D_MODEL + h * HEAD_DIM + d;
                            dot += q_t->data[qi] * k_t->data[ki];
                        }
                        scores[p] = dot * scale;
                        if (scores[p] > max_s) max_s = scores[p];
                    }

                    // Softmax
                    float sum_exp = 0.0f;
                    for (int p = 0; p < N_PATCHES; p++) {
                        scores[p] = expf(scores[p] - max_s);
                        sum_exp += scores[p];
                    }
                    for (int p = 0; p < N_PATCHES; p++)
                        scores[p] /= (sum_exp + 1e-8f);

                    // Weighted sum of values
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float val = 0.0f;
                        for (int p = 0; p < N_PATCHES; p++) {
                            val += scores[p] *
                                v_t->data[p * D_MODEL + h * HEAD_DIM + d];
                        }
                        out->data[t * D_MODEL + h * HEAD_DIM + d] = val;
                    }
                }
            }

            int cross_out_idx = nt_tape_get()->count;
            nt_tape_get()->entries[cross_out_idx].output = out;
            nt_tape_get()->entries[cross_out_idx].op = NT_OP_NONE;
            nt_tape_get()->count++;

            int co_proj = nt_seq_linear(w->layers[l].co,
                                         cross_out_idx, T);
            x = nt_add(x, co_proj);
        }
        x = nt_seq_layernorm(x, w->layers[l].ln2_g,
                              w->layers[l].ln2_b, T, D_MODEL);

        // FFN
        int ff = nt_seq_linear(w->layers[l].ff1, x, T);
        ff = nt_gelu(ff);
        ff = nt_seq_linear(w->layers[l].ff2, ff, T);
        x = nt_add(x, ff);
        x = nt_seq_layernorm(x, w->layers[l].ln3_g,
                              w->layers[l].ln3_b, T, D_MODEL);
    }

    // Final layer norm
    x = nt_seq_layernorm(x, w->ln_final_g, w->ln_final_b, T, D_MODEL);

    // Output projection (weight-tied with wte)
    int logits = nt_seq_linear(w->wte, x, T);

    return logits;
}

// ── Save weights ─────────────────────────────────────────────────────────────

static void save_weights(const char* path) {
    // Collect all param tensors from the tape
    nt_tape* tape = nt_tape_get();
    int np = tape->n_params;
    nt_tensor** params = (nt_tensor**)malloc(np * sizeof(nt_tensor*));
    if (!params) { printf("  Failed to allocate param array\n"); return; }

    for (int i = 0; i < np; i++) {
        params[i] = tape->entries[i].output;
    }

    int ret = nt_save(path, params, np);
    if (ret == 0) {
        printf("  Weights saved to %s (%d params)\n", path, np);
    } else {
        printf("  Failed to save: %s\n", path);
    }
    free(params);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(void) {
    printf("==========================================================\n");
    printf("  train_vlm.c — VLM training in pure C\n");
    printf("  Engine: notorch | Optimizer: Chuck\n");
    printf("==========================================================\n\n");

    nt_seed(42);

    // Training text (English)
    const char* text =
        "This is a red square. "
        "The image shows a red object in the center. "
        "A red square is located in the middle. "
        "The object in the image is red. "
        "There is a red square in the center of the image. "
        "The central area contains a red square shape. ";

    init_vocab(text);
    int text_len = (int)strlen(text);

    // Encode text
    int* token_ids = (int*)malloc(text_len * sizeof(int));
    for (int i = 0; i < text_len; i++) {
        token_ids[i] = encode_char(text[i]);
    }

    // Create synthetic image features
    float image_features[N_PATCHES * IMAGE_DIM];
    for (int p = 0; p < N_PATCHES; p++) {
        create_synthetic_image(
            &image_features[p * IMAGE_DIM], IMAGE_DIM);
    }

    // Initialize model
    VLMWeights w = init_model();

    printf("  Architecture:\n");
    printf("   d_model=%d, heads=%d, layers=%d, ff=%d\n",
           D_MODEL, N_HEADS, N_LAYERS, D_FF);
    printf("   vocab=%d, max_seq=%d, patches=%d\n",
           vocab_len, MAX_SEQ, N_PATCHES);
    printf("   Total parameters: %d\n\n", w.total_params);

    // Training loop
    printf("  Training with Chuck optimizer (lr=%.4f)\n", LR);
    printf("  ──────────────────────────────────────\n");

    nt_train_mode(1);
    float best_loss = 1e30f;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Random sequence window
        int max_start = text_len - MAX_SEQ - 1;
        if (max_start < 0) max_start = 0;
        int start = (int)(vlm_rand_uniform() * max_start);
        int T = MAX_SEQ;
        if (start + T + 1 > text_len) T = text_len - start - 1;
        if (T < 2) continue;

        // Prepare input and target tokens
        int* input_tokens = &token_ids[start];
        int* target_tokens = &token_ids[start + 1];

        // Forward pass
        int logits_idx = forward_pass(&w, input_tokens, T,
                                       image_features);

        // Cross-entropy loss
        nt_tensor* targets = nt_tensor_new(T);
        for (int i = 0; i < T; i++)
            targets->data[i] = (float)target_tokens[i];
        int tgt_idx = nt_tape_get()->count;
        nt_tape_get()->entries[tgt_idx].output = targets;
        nt_tape_get()->entries[tgt_idx].op = NT_OP_NONE;
        nt_tape_get()->count++;

        int loss_idx = nt_seq_cross_entropy(logits_idx, tgt_idx,
                                             T, vocab_len);
        float loss_val = nt_tape_get()->entries[loss_idx].output->data[0];

        // Backward + Chuck step
        nt_tape_backward(loss_idx);
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(LR, loss_val);

        if (loss_val < best_loss) best_loss = loss_val;

        if (epoch % 20 == 0 || epoch == EPOCHS - 1) {
            printf("   epoch %3d | loss %.4f | best %.4f\n",
                   epoch, loss_val, best_loss);
        }

        // Clear tape for next iteration (keep params)
        // Reset computation graph but preserve parameters
        for (int i = 0; i < nt_tape_get()->count; i++) {
            if (nt_tape_get()->entries[i].grad) {
                nt_tensor_free(nt_tape_get()->entries[i].grad);
                nt_tape_get()->entries[i].grad = NULL;
            }
        }
        nt_tape_get()->count = nt_tape_get()->n_params;
    }

    printf("\n  Training complete.\n");
    printf("   Final best loss: %.4f\n", best_loss);

    // Save weights
    save_weights(SAVE_PATH);

    // Cleanup
    free(token_ids);
    nt_tape_clear();

    printf("\n  Done. Resonance is unbreakable.\n");
    return 0;
}
