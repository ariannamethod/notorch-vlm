// serve.c — HTTP streaming inference server for simple_vlm
// Part of simple_vlm project (Arianna Method)
//
// Minimal HTTP server: serves index.html + SSE endpoint for token streaming.
// No dependencies beyond POSIX sockets. Opens in any browser.
//
// Build:
//   cc -std=c11 -O2 -I. -o serve serve.c notorch.c -lm -lpthread
//
// Run:
//   ./serve [port] [weights_path]
//   # defaults: port=8080, weights=../weights/vlm_notorch.bin
//   # then open http://localhost:8080 in browser

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <errno.h>

// ── Model config (must match training) ────────────────────────────────────────

// These will be overridden by config presets
static int    CFG_D_MODEL   = 160;
static int    CFG_N_HEADS   = 8;
static int    CFG_N_LAYERS  = 4;
static int    CFG_D_FF      = 496;
static int    CFG_MAX_SEQ   = 128;
static int    CFG_N_PATCHES = 16;
static int    CFG_IMAGE_DIM = 64;
static int    CFG_VOCAB     = 33;  // updated from weights

#define MAX_RESPONSE_TOKENS 256
#define SERVER_BACKLOG      8

// ── Vocab (character-level) ───────────────────────────────────────────────────

// Default character vocab (space + lowercase + punctuation)
// Will be rebuilt from training data
static char  g_vocab[256];
static int   g_vocab_len = 0;
static int   g_char_to_id[256];

static void init_default_vocab(void) {
    // Matches the training text vocab: space, '.', letters, punctuation
    const char* chars = " ,.ABILOTWabc"
                        "defghijklmnopqrstuvwxy";
    memset(g_char_to_id, 0, sizeof(g_char_to_id));
    g_vocab_len = 0;
    for (int i = 0; chars[i]; i++) {
        g_vocab[g_vocab_len] = chars[i];
        g_char_to_id[(unsigned char)chars[i]] = g_vocab_len;
        g_vocab_len++;
    }
    CFG_VOCAB = g_vocab_len;
    printf("  Vocab: %d characters\n", g_vocab_len);
}

static int encode_char(char c) {
    int id = g_char_to_id[(unsigned char)c];
    return id;
}

// ── Synthetic image features ──────────────────────────────────────────────────

static float g_image_features[16 * 64];  // N_PATCHES * IMAGE_DIM max

static void create_image_features(void) {
    int len = CFG_N_PATCHES * CFG_IMAGE_DIM;
    for (int i = 0; i < len; i++) g_image_features[i] = 0.1f;
    for (int p = 0; p < CFG_N_PATCHES; p++) {
        int base = p * CFG_IMAGE_DIM;
        for (int i = 0; i < CFG_IMAGE_DIM / 4; i++)
            g_image_features[base + i] = 0.8f;
        for (int i = CFG_IMAGE_DIM / 4; i < CFG_IMAGE_DIM / 2; i++)
            g_image_features[base + i] = 0.6f;
        for (int i = CFG_IMAGE_DIM / 2; i < 3 * CFG_IMAGE_DIM / 4; i++)
            g_image_features[base + i] = 0.4f;
    }
}

// ── Model state ───────────────────────────────────────────────────────────────

static nt_tensor** g_params = NULL;
static int         g_n_params = 0;

static int load_model(const char* path) {
    printf("  Loading weights from %s...\n", path);
    g_params = nt_load(path, &g_n_params);
    if (!g_params) {
        fprintf(stderr, "  ERROR: Failed to load weights from %s\n", path);
        return -1;
    }
    long total = nt_count_params(g_params, g_n_params);
    printf("  Loaded %d param tensors (%ld total floats)\n", g_n_params, total);
    return 0;
}

// ── Forward pass (inference only) ─────────────────────────────────────────────

static float* inference_logits(int* tokens, int T) {
    // Register params on tape
    nt_tape_start();
    nt_train_mode(0);

    int* tape_ids = (int*)malloc(g_n_params * sizeof(int));
    for (int i = 0; i < g_n_params; i++) {
        tape_ids[i] = nt_tape_param(g_params[i]);
    }

    int D = CFG_D_MODEL;
    int HD = D / CFG_N_HEADS;
    int NP = CFG_N_PATCHES;

    int pi = 0;

    // Vision
    int vis_pos_i = tape_ids[pi++];
    int vis_w     = tape_ids[pi++];
    int wte_i     = tape_ids[pi++];
    int wpe_i     = tape_ids[pi++];

    // Image features as tape entry
    nt_tensor* img = nt_tensor_new(NP * CFG_IMAGE_DIM);
    for (int i = 0; i < NP * CFG_IMAGE_DIM; i++)
        img->data[i] = g_image_features[i];
    int img_idx = nt_tape_get()->count;
    nt_tape_get()->entries[img_idx].output = nt_tensor_ref(img);
    nt_tape_get()->entries[img_idx].op = NT_OP_NONE;
    nt_tape_get()->count++;
    nt_tensor_free(img);

    int vis = nt_seq_linear(vis_w, img_idx, NP);
    vis = nt_add(vis, vis_pos_i);

    // Tokens
    nt_tensor* tok = nt_tensor_new(T);
    for (int i = 0; i < T; i++) tok->data[i] = (float)tokens[i];
    int tok_idx = nt_tape_get()->count;
    nt_tape_get()->entries[tok_idx].output = nt_tensor_ref(tok);
    nt_tape_get()->entries[tok_idx].op = NT_OP_NONE;
    nt_tape_get()->count++;
    nt_tensor_free(tok);

    int h = nt_seq_embedding(wte_i, wpe_i, tok_idx, T, D);

    // Transformer layers
    for (int l = 0; l < CFG_N_LAYERS; l++) {
        int wq = tape_ids[pi++]; int wk = tape_ids[pi++];
        int wv = tape_ids[pi++]; int wo = tape_ids[pi++];
        int ln1g = tape_ids[pi++]; int ln1b = tape_ids[pi++];
        int cq = tape_ids[pi++]; int ck = tape_ids[pi++];
        int cv = tape_ids[pi++]; int co = tape_ids[pi++];
        int ln2g = tape_ids[pi++]; int ln2b = tape_ids[pi++];
        int ff1 = tape_ids[pi++]; int ff2 = tape_ids[pi++];
        int ln3g = tape_ids[pi++]; int ln3b = tape_ids[pi++];

        // Self-attention
        int q = nt_seq_linear(wq, h, T);
        int k = nt_seq_linear(wk, h, T);
        int v = nt_seq_linear(wv, h, T);
        int attn = nt_mh_causal_attention(q, k, v, T, HD);
        attn = nt_seq_linear(wo, attn, T);
        h = nt_add(h, attn);
        h = nt_seq_layernorm(h, ln1g, ln1b, T, D);

        // Cross-attention
        int cq_out = nt_seq_linear(cq, h, T);
        int ck_out = nt_seq_linear(ck, vis, NP);
        int cv_out = nt_seq_linear(cv, vis, NP);
        int cross = nt_mh_cross_attention(cq_out, ck_out, cv_out, T, NP, HD);
        int co_out = nt_seq_linear(co, cross, T);
        h = nt_add(h, co_out);
        h = nt_seq_layernorm(h, ln2g, ln2b, T, D);

        // FFN
        int f = nt_seq_linear(ff1, h, T);
        f = nt_gelu(f);
        f = nt_seq_linear(ff2, f, T);
        h = nt_add(h, f);
        h = nt_seq_layernorm(h, ln3g, ln3b, T, D);
    }

    // Final norm + head
    int lnfg = tape_ids[pi++];
    int lnfb = tape_ids[pi++];
    int head_w = tape_ids[pi++];

    h = nt_seq_layernorm(h, lnfg, lnfb, T, D);
    int logits_idx = nt_seq_linear(head_w, h, T);

    // Extract last-position logits
    nt_tensor* logits_t = nt_tape_get()->entries[logits_idx].output;
    int V = CFG_VOCAB;
    int offset = (T - 1) * V;
    float* result = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) result[i] = logits_t->data[offset + i];

    free(tape_ids);
    nt_tape_clear();
    return result;
}

// ── Sampling ──────────────────────────────────────────────────────────────────

static int sample_token(float* logits, int V, float temperature, int top_k) {
    // Temperature scaling
    for (int i = 0; i < V; i++) logits[i] /= temperature;

    // Top-k filtering
    if (top_k > 0 && top_k < V) {
        // Find k-th largest
        float* sorted = (float*)malloc(V * sizeof(float));
        memcpy(sorted, logits, V * sizeof(float));
        // Simple selection for top-k threshold
        for (int i = 0; i < top_k; i++) {
            for (int j = i + 1; j < V; j++) {
                if (sorted[j] > sorted[i]) {
                    float tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
                }
            }
        }
        float threshold = sorted[top_k - 1];
        free(sorted);
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) logits[i] = -1e30f;
        }
    }

    // Softmax
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0;
    for (int i = 0; i < V; i++) { logits[i] = expf(logits[i] - mx); sum += logits[i]; }
    for (int i = 0; i < V; i++) logits[i] /= sum;

    // Multinomial sample
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    for (int i = 0; i < V; i++) {
        cum += logits[i];
        if (cum >= r) return i;
    }
    return V - 1;
}

// ── Embedded HTML ─────────────────────────────────────────────────────────────

static const char* HTML_PAGE =
"<!DOCTYPE html>\n"
"<html><head><meta charset='utf-8'>\n"
"<title>simple_vlm — live inference</title>\n"
"<style>\n"
"  body { background: #0d1117; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; max-width: 720px; margin: 40px auto; padding: 0 20px; }\n"
"  h1 { color: #58a6ff; font-size: 1.4em; }\n"
"  .info { color: #8b949e; font-size: 0.85em; margin-bottom: 20px; }\n"
"  #output { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; min-height: 120px; white-space: pre-wrap; font-size: 1.1em; line-height: 1.6; }\n"
"  .token { color: #f0f6fc; }\n"
"  .cursor { background: #58a6ff; color: #0d1117; animation: blink 1s step-end infinite; }\n"
"  @keyframes blink { 50%% { opacity: 0; } }\n"
"  .controls { margin: 16px 0; display: flex; gap: 10px; align-items: center; }\n"
"  input[type=text] { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; padding: 8px 12px; border-radius: 6px; flex: 1; font-family: inherit; }\n"
"  button { background: #238636; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-family: inherit; }\n"
"  button:hover { background: #2ea043; }\n"
"  button:disabled { background: #30363d; cursor: default; }\n"
"  .label { color: #8b949e; font-size: 0.8em; }\n"
"  select { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 8px; border-radius: 4px; }\n"
"  .footer { margin-top: 20px; color: #484f58; font-size: 0.75em; text-align: center; }\n"
"</style>\n"
"</head><body>\n"
"<h1>&#x1f441; simple_vlm — streaming inference</h1>\n"
"<div class='info'>notorch + Chuck | pure C engine | no PyTorch, no numpy</div>\n"
"<div class='controls'>\n"
"  <input type='text' id='prompt' placeholder='describe the image...' value='The image shows'>\n"
"  <select id='temp'><option value='0.5'>temp=0.5</option><option value='0.8' selected>temp=0.8</option><option value='1.0'>temp=1.0</option></select>\n"
"  <button id='btn' onclick='generate()'>Generate</button>\n"
"</div>\n"
"<div id='output'><span class='cursor'>&nbsp;</span></div>\n"
"<div class='footer'>Arianna Method &mdash; resonance is unbreakable</div>\n"
"<script>\n"
"let evtSource = null;\n"
"function generate() {\n"
"  const prompt = document.getElementById('prompt').value;\n"
"  const temp = document.getElementById('temp').value;\n"
"  const btn = document.getElementById('btn');\n"
"  const out = document.getElementById('output');\n"
"  btn.disabled = true;\n"
"  out.innerHTML = '<span class=\"token\">' + escHtml(prompt) + '</span><span class=\"cursor\">&nbsp;</span>';\n"
"  if (evtSource) evtSource.close();\n"
"  evtSource = new EventSource('/generate?prompt=' + encodeURIComponent(prompt) + '&temp=' + temp);\n"
"  evtSource.onmessage = function(e) {\n"
"    if (e.data === '[DONE]') { evtSource.close(); btn.disabled = false; out.querySelector('.cursor').remove(); return; }\n"
"    const cursor = out.querySelector('.cursor');\n"
"    const span = document.createElement('span');\n"
"    span.className = 'token';\n"
"    span.textContent = e.data;\n"
"    out.insertBefore(span, cursor);\n"
"    out.scrollTop = out.scrollHeight;\n"
"  };\n"
"  evtSource.onerror = function() { evtSource.close(); btn.disabled = false; };\n"
"}\n"
"function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }\n"
"document.getElementById('prompt').addEventListener('keydown', function(e) { if (e.key === 'Enter') generate(); });\n"
"</script>\n"
"</body></html>\n";

// ── HTTP helpers ──────────────────────────────────────────────────────────────

static void send_str(int fd, const char* s) {
    int len = (int)strlen(s);
    int sent = 0;
    while (sent < len) {
        int n = write(fd, s + sent, len - sent);
        if (n <= 0) break;
        sent += n;
    }
}

static void send_http_response(int fd, const char* status, const char* content_type,
                                const char* body) {
    char header[512];
    snprintf(header, sizeof(header),
             "HTTP/1.1 %s\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %d\r\n"
             "Connection: close\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "\r\n", status, content_type, (int)strlen(body));
    send_str(fd, header);
    send_str(fd, body);
}

// ── SSE streaming generate ────────────────────────────────────────────────────

static void handle_generate(int fd, const char* prompt, float temperature) {
    // SSE headers
    char header[256];
    snprintf(header, sizeof(header),
             "HTTP/1.1 200 OK\r\n"
             "Content-Type: text/event-stream\r\n"
             "Cache-Control: no-cache\r\n"
             "Connection: keep-alive\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "\r\n");
    send_str(fd, header);

    // Encode prompt
    int prompt_len = (int)strlen(prompt);
    int* ctx = (int*)malloc((prompt_len + MAX_RESPONSE_TOKENS) * sizeof(int));
    int ctx_len = 0;
    for (int i = 0; i < prompt_len && ctx_len < CFG_MAX_SEQ; i++) {
        ctx[ctx_len++] = encode_char(prompt[i]);
    }

    // Generate tokens one by one, streaming via SSE
    for (int step = 0; step < MAX_RESPONSE_TOKENS; step++) {
        int T = ctx_len;
        if (T > CFG_MAX_SEQ) {
            // Shift window
            int shift = T - CFG_MAX_SEQ;
            memmove(ctx, ctx + shift, CFG_MAX_SEQ * sizeof(int));
            ctx_len = CFG_MAX_SEQ;
            T = CFG_MAX_SEQ;
        }

        float* logits = inference_logits(ctx, T);
        if (!logits) break;

        int next_id = sample_token(logits, CFG_VOCAB, temperature, 20);
        free(logits);

        // Decode token to character
        char ch = (next_id >= 0 && next_id < g_vocab_len) ? g_vocab[next_id] : '?';

        // Send SSE event
        char event[64];
        if (ch == '\n') {
            snprintf(event, sizeof(event), "data: \\n\n\n");
        } else {
            snprintf(event, sizeof(event), "data: %c\n\n", ch);
        }
        int n = write(fd, event, strlen(event));
        if (n <= 0) break;  // client disconnected

        ctx[ctx_len++] = next_id;
    }

    // Send done marker
    send_str(fd, "data: [DONE]\n\n");
    free(ctx);
}

// ── URL decoding ──────────────────────────────────────────────────────────────

static void url_decode(char* dst, const char* src, int max_len) {
    int di = 0;
    for (int i = 0; src[i] && di < max_len - 1; i++) {
        if (src[i] == '%' && src[i+1] && src[i+2]) {
            char hex[3] = {src[i+1], src[i+2], 0};
            dst[di++] = (char)strtol(hex, NULL, 16);
            i += 2;
        } else if (src[i] == '+') {
            dst[di++] = ' ';
        } else {
            dst[di++] = src[i];
        }
    }
    dst[di] = 0;
}

// ── Request handler ───────────────────────────────────────────────────────────

static void* handle_client(void* arg) {
    int fd = *(int*)arg;
    free(arg);

    char buf[4096];
    int n = read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(fd); return NULL; }
    buf[n] = 0;

    // Parse request line
    char method[16], path[2048];
    sscanf(buf, "%15s %2047s", method, path);

    if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) {
        // Serve the embedded HTML
        send_http_response(fd, "200 OK", "text/html; charset=utf-8", HTML_PAGE);
    }
    else if (strncmp(path, "/generate?", 10) == 0) {
        // Parse query params
        char prompt_raw[1024] = "The image shows";
        float temp = 0.8f;

        char* q = path + 10;
        char* tok = strtok(q, "&");
        while (tok) {
            if (strncmp(tok, "prompt=", 7) == 0) {
                url_decode(prompt_raw, tok + 7, sizeof(prompt_raw));
            } else if (strncmp(tok, "temp=", 5) == 0) {
                temp = atof(tok + 5);
                if (temp < 0.1f) temp = 0.1f;
                if (temp > 2.0f) temp = 2.0f;
            }
            tok = strtok(NULL, "&");
        }

        handle_generate(fd, prompt_raw, temp);
    }
    else if (strcmp(path, "/health") == 0) {
        char body[256];
        snprintf(body, sizeof(body),
                 "{\"status\":\"ok\",\"params\":%d,\"vocab\":%d,\"d_model\":%d}",
                 g_n_params, CFG_VOCAB, CFG_D_MODEL);
        send_http_response(fd, "200 OK", "application/json", body);
    }
    else {
        send_http_response(fd, "404 Not Found", "text/plain", "not found");
    }

    close(fd);
    return NULL;
}

// ── Main ──────────────────────────────────────────────────────────────────────

static volatile int g_running = 1;

static void sigint_handler(int sig) {
    (void)sig;
    g_running = 0;
}

int main(int argc, char** argv) {
    int port = 8080;
    const char* weights_path = "../weights/vlm_notorch.bin";

    if (argc >= 2) port = atoi(argv[1]);
    if (argc >= 3) weights_path = argv[2];

    printf("==========================================================\n");
    printf("  simple_vlm — streaming inference server\n");
    printf("  Engine: notorch | Optimizer: Chuck\n");
    printf("  No PyTorch. No numpy. Pure C.\n");
    printf("==========================================================\n\n");

    nt_seed(42);
    init_default_vocab();
    create_image_features();

    if (load_model(weights_path) != 0) {
        fprintf(stderr, "  Failed to load model. Run train.py first.\n");
        return 1;
    }

    // Create socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, SERVER_BACKLOG) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    signal(SIGINT, sigint_handler);
    signal(SIGPIPE, SIG_IGN);

    printf("\n  Server running on http://localhost:%d\n", port);
    printf("  Open in browser to try streaming inference.\n");
    printf("  Press Ctrl+C to stop.\n\n");

    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        int* fd_ptr = (int*)malloc(sizeof(int));
        *fd_ptr = client_fd;

        pthread_t tid;
        if (pthread_create(&tid, NULL, handle_client, fd_ptr) != 0) {
            close(client_fd);
            free(fd_ptr);
        } else {
            pthread_detach(tid);
        }
    }

    close(server_fd);
    printf("\n  Server stopped.\n");
    return 0;
}
