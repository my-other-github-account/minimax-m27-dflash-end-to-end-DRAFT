// dump_hiddens.cpp - capture target model hidden states at DFlash layers
// for direct comparison with the FP8 training traces.
//
// Usage:
//   ./dump_hiddens -m TARGET.gguf --token-ids-file <bin file with i64 tokens> -o out.bin
// Output binary format (little-endian):
//   [n_layers:i32][n_tokens:i32][n_embd:i32]
//   followed by f32 hidden states for each layer in order:
//   layer_0: [n_tokens, n_embd]
//   layer_1: [n_tokens, n_embd]
//   ...
//
// Linking: links against libllama.so + libcommon (header-only common args parsing okay too).

#include "llama.h"
#include "ggml.h"
#include "common.h"
#include "arg.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "usage: %s -m TARGET.gguf -p PROMPT -o OUT.bin\n", argv[0]);
        return 1;
    }

    std::string out_path;
    const char * envp = getenv("OUT_BIN");
    if (envp) out_path = envp;
    else out_path = "/tmp/buun_hiddens.bin";

    // Optional: read tokens from a binary file instead of tokenizing the prompt.
    // Format: [n_tokens:u32][n_tokens × i32]
    std::vector<llama_token> override_tokens;
    const char * tok_path = getenv("TOKENS_BIN");
    if (tok_path) {
        std::ifstream tf(tok_path, std::ios::binary);
        if (!tf) { fprintf(stderr, "cannot open TOKENS_BIN %s\n", tok_path); return 1; }
        uint32_t n = 0; tf.read((char*)&n, sizeof(n));
        override_tokens.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            int32_t t; tf.read((char*)&t, sizeof(t));
            override_tokens[i] = (llama_token)t;
        }
        fprintf(stderr, "loaded %u tokens from %s\n", n, tok_path);
    }

    llama_backend_init();
    common_init();

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    auto cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to init ctx\n"); return 1; }

    // Read target layer ids: default same as training, override with CAPTURE_LAYERS env (CSV).
    std::vector<int32_t> capture_layers = {2, 16, 30, 45, 59};
    const char * cl_env = getenv("CAPTURE_LAYERS");
    if (cl_env) {
        capture_layers.clear();
        std::string s = cl_env;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos) comma = s.size();
            try { capture_layers.push_back(std::stoi(s.substr(pos, comma - pos))); } catch (...) {}
            pos = comma + 1;
        }
        fprintf(stderr, "CAPTURE_LAYERS override: ");
        for (auto l : capture_layers) fprintf(stderr, "%d ", l);
        fprintf(stderr, "\n");
    }
    llama_set_dflash_capture(ctx, capture_layers.data(), (int32_t)capture_layers.size());

    // Tokenize the prompt unless we have override tokens
    std::vector<llama_token> tokens;
    if (!override_tokens.empty()) {
        tokens = override_tokens;
    } else {
        tokens = common_tokenize(ctx, params.prompt, true);
    }
    fprintf(stderr, "n_tokens = %zu\n", tokens.size());

    // Single batch decode
    llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], (llama_pos)i, {0}, /*logits=*/true);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    llama_batch_free(batch);

    int32_t n_slots = llama_get_n_layer_hiddens(ctx);
    fprintf(stderr, "n_slots = %d\n", n_slots);
    if (n_slots == 0) {
        fprintf(stderr, "no hidden state slots captured!\n");
        return 1;
    }

    int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx, 0);
    int64_t n_embd   = llama_get_layer_hidden_n_embd(ctx, 0);
    fprintf(stderr, "n_tokens=%ld n_embd=%ld\n", n_tokens, n_embd);

    std::ofstream out(out_path, std::ios::binary);
    int32_t hdr[3] = {n_slots, (int32_t)n_tokens, (int32_t)n_embd};
    out.write((char*)hdr, sizeof(hdr));
    // Write capture_layers as i32 list of length n_slots
    out.write((char*)capture_layers.data(), (size_t)n_slots * sizeof(int32_t));
    // Also dump token ids for sanity
    int32_t n_toks_in = (int32_t)tokens.size();
    out.write((char*)&n_toks_in, sizeof(n_toks_in));
    std::vector<int32_t> tok_i32(tokens.begin(), tokens.end());
    out.write((char*)tok_i32.data(), tok_i32.size() * sizeof(int32_t));

    for (int slot = 0; slot < n_slots; ++slot) {
        float * data = llama_get_layer_hidden(ctx, slot);
        if (!data) { fprintf(stderr, "slot %d returned null\n", slot); continue; }
        int64_t nt = llama_get_layer_hidden_n_tokens(ctx, slot);
        int64_t ne = llama_get_layer_hidden_n_embd(ctx, slot);
        fprintf(stderr, "slot %d: layer %d, n_tokens=%ld n_embd=%ld\n", slot, capture_layers[slot], nt, ne);
        out.write((char*)data, nt * ne * sizeof(float));
    }
    out.close();
    fprintf(stderr, "wrote %s\n", out_path.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
