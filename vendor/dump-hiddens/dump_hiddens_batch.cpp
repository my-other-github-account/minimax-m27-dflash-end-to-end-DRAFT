// dump_hiddens_batch.cpp — capture target model hidden states for MANY samples
// in a single process invocation. Reads a manifest of (input_bin, output_bin)
// pairs and processes each sequentially with a single context.
//
// Manifest format: one line per sample, "<tokens_bin> <out_bin>\n"
//
// Each tokens_bin is [u32 n_tokens][i32 token_ids...] (same as TOKENS_BIN env).
// Each out_bin is the same format as the original dump_hiddens.cpp:
//   [n_layers:i32][n_tokens:i32][n_embd:i32]
//   [capture_layers: i32 × n_layers]
//   [n_toks_in:i32][token_ids: i32 × n_toks_in]
//   [hidden f32: n_layers × n_tokens × n_embd]
//
// CAPTURE_LAYERS env (CSV) selects which residuals to grab. Defaults to
// 2,16,30,45,59,61 for MiniMax-M2.7 5L+last-residual recipe.
//
// MANIFEST env points at the manifest file.
//
// Usage: same llama args as dump-hiddens, plus -p anything (ignored when
// manifest is set). Example:
//   MANIFEST=/tmp/m.txt CAPTURE_LAYERS=2,16,30,45,59,61 \
//     ./llama-dump-hiddens-batch -m IQ4.gguf -ngl 99 -ot exps=CPU -c 8192 -p x

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
#include <chrono>

static std::vector<int32_t> parse_csv_i32(const char * s) {
    std::vector<int32_t> out;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        if (comma == std::string::npos) comma = str.size();
        try { out.push_back(std::stoi(str.substr(pos, comma - pos))); } catch (...) {}
        pos = comma + 1;
    }
    return out;
}

static bool read_tokens_bin(const std::string & path, std::vector<llama_token> & out) {
    std::ifstream tf(path, std::ios::binary);
    if (!tf) return false;
    uint32_t n = 0; tf.read((char*)&n, sizeof(n));
    out.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        int32_t t; tf.read((char*)&t, sizeof(t));
        out[i] = (llama_token)t;
    }
    return tf.good() || tf.eof();
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "usage: MANIFEST=m.txt %s -m TARGET.gguf [-c N] -ngl 99\n", argv[0]);
        return 1;
    }

    const char * man_path = getenv("MANIFEST");
    if (!man_path) {
        fprintf(stderr, "ERR: set MANIFEST=<path-to-manifest>\n");
        return 1;
    }

    // Parse capture layers (CSV) once.
    std::vector<int32_t> capture_layers = {2, 16, 30, 45, 59, 61};
    if (const char * cl = getenv("CAPTURE_LAYERS")) {
        capture_layers = parse_csv_i32(cl);
    }
    fprintf(stderr, "capture_layers: ");
    for (auto l : capture_layers) fprintf(stderr, "%d ", l);
    fprintf(stderr, "\n");

    // Read manifest
    std::vector<std::pair<std::string,std::string>> jobs;
    {
        std::ifstream mf(man_path);
        if (!mf) { fprintf(stderr, "cannot open manifest %s\n", man_path); return 1; }
        std::string line;
        while (std::getline(mf, line)) {
            if (line.empty()) continue;
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            jobs.emplace_back(line.substr(0, sp), line.substr(sp+1));
        }
    }
    fprintf(stderr, "manifest: %zu jobs\n", jobs.size());
    if (jobs.empty()) return 0;

    llama_backend_init();
    common_init();

    auto mparams = common_model_params_to_llama(params);
    auto t_load0 = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    auto t_load1 = std::chrono::steady_clock::now();
    fprintf(stderr, "model loaded in %.1fs\n",
            std::chrono::duration<double>(t_load1 - t_load0).count());

    auto cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to init ctx\n"); return 1; }

    llama_set_dflash_capture(ctx, capture_layers.data(), (int32_t)capture_layers.size());

    int n_ok = 0, n_fail = 0;
    auto t_run0 = std::chrono::steady_clock::now();

    for (size_t j = 0; j < jobs.size(); ++j) {
        const std::string & tok_path = jobs[j].first;
        const std::string & out_path = jobs[j].second;

        std::vector<llama_token> tokens;
        if (!read_tokens_bin(tok_path, tokens) || tokens.empty()) {
            fprintf(stderr, "[%zu] cannot read %s\n", j, tok_path.c_str());
            n_fail++;
            continue;
        }

        // Clear KV cache between samples (each sample is an independent prompt)
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) llama_memory_clear(mem, true);

        // Single-batch decode of the prompt
        llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], (llama_pos)i, {0}, /*logits=*/true);
        }
        auto t0 = std::chrono::steady_clock::now();
        int rc = llama_decode(ctx, batch);
        auto t1 = std::chrono::steady_clock::now();
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "[%zu] decode failed rc=%d ntok=%zu\n", j, rc, tokens.size());
            n_fail++;
            continue;
        }

        int32_t n_slots = llama_get_n_layer_hiddens(ctx);
        if (n_slots == 0) {
            fprintf(stderr, "[%zu] no slots captured\n", j);
            n_fail++;
            continue;
        }
        int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx, 0);
        int64_t n_embd   = llama_get_layer_hidden_n_embd(ctx, 0);

        std::ofstream out(out_path, std::ios::binary);
        if (!out) { fprintf(stderr, "[%zu] cannot open out %s\n", j, out_path.c_str()); n_fail++; continue; }
        int32_t hdr[3] = {n_slots, (int32_t)n_tokens, (int32_t)n_embd};
        out.write((char*)hdr, sizeof(hdr));
        out.write((char*)capture_layers.data(), (size_t)n_slots * sizeof(int32_t));
        int32_t n_toks_in = (int32_t)tokens.size();
        out.write((char*)&n_toks_in, sizeof(n_toks_in));
        std::vector<int32_t> tok_i32(tokens.begin(), tokens.end());
        out.write((char*)tok_i32.data(), tok_i32.size() * sizeof(int32_t));
        for (int slot = 0; slot < n_slots; ++slot) {
            float * data = llama_get_layer_hidden(ctx, slot);
            int64_t nt = llama_get_layer_hidden_n_tokens(ctx, slot);
            int64_t ne = llama_get_layer_hidden_n_embd(ctx, slot);
            if (!data || nt != n_tokens || ne != n_embd) {
                fprintf(stderr, "[%zu] slot %d shape mismatch: data=%p nt=%ld ne=%ld\n",
                        j, slot, (void*)data, nt, ne);
                n_fail++;
                out.close();
                std::remove(out_path.c_str());
                goto next_job;
            }
            out.write((char*)data, nt * ne * sizeof(float));
        }
        out.close();
        {
            double sec = std::chrono::duration<double>(t1 - t0).count();
            fprintf(stderr, "[%zu] OK ntok=%ld decode=%.2fs -> %s\n",
                    j, n_tokens, sec, out_path.c_str());
        }
        n_ok++;

        // periodic progress + cumulative throughput
        if ((j+1) % 10 == 0) {
            auto tnow = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(tnow - t_run0).count();
            fprintf(stderr, "[progress] %zu/%zu ok=%d fail=%d elapsed=%.1fs avg=%.2fs/sample\n",
                    j+1, jobs.size(), n_ok, n_fail, elapsed, elapsed / (j+1));
        }

        next_job: ;
    }

    auto t_run1 = std::chrono::steady_clock::now();
    fprintf(stderr, "DONE ok=%d fail=%d total=%zu in %.1fs\n",
            n_ok, n_fail, jobs.size(),
            std::chrono::duration<double>(t_run1 - t_run0).count());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return n_fail == 0 ? 0 : 2;
}
