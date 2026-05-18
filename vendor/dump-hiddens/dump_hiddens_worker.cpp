// dump_hiddens_worker.cpp — persistent stdin/stdout worker for trace generation.
//
// Protocol:
//   stdout once at startup:  READY\t<ctx>\n
//   stdin per request:       <req_id>\t<tokens_bin>\t<out_bin>\t<layer_csv>\n
//   stdout success:          OK\t<req_id>\t<n_layers>\t<n_tokens>\t<n_embd>\n
//   stdout failure:          ERR\t<req_id>\t<message>\n
//
// All logs stay on stderr so stdout is reserved for protocol lines.

#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> parse_csv_i32(const std::string & s) {
    std::vector<int32_t> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) {
            comma = s.size();
        }
        try {
            out.push_back(std::stoi(s.substr(pos, comma - pos)));
        } catch (...) {
        }
        pos = comma + 1;
    }
    return out;
}

static bool read_tokens_bin(const std::string & path, std::vector<llama_token> & out) {
    std::ifstream tf(path, std::ios::binary);
    if (!tf) {
        return false;
    }
    uint32_t n = 0;
    tf.read((char *) &n, sizeof(n));
    out.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        int32_t t = 0;
        tf.read((char *) &t, sizeof(t));
        out[i] = (llama_token) t;
    }
    return tf.good() || tf.eof();
}

static bool write_hidden_bin(
    llama_context * ctx,
    const std::string & out_path,
    const std::vector<int32_t> & capture_layers,
    const std::vector<llama_token> & tokens,
    int32_t & n_slots_out,
    int32_t & n_tokens_out,
    int32_t & n_embd_out,
    std::string & err
) {
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        err = "cannot open output path";
        return false;
    }

    int32_t n_slots = llama_get_n_layer_hiddens(ctx);
    if (n_slots == 0) {
        err = "no hidden state slots captured";
        out.close();
        std::remove(out_path.c_str());
        return false;
    }

    int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx, 0);
    int64_t n_embd = llama_get_layer_hidden_n_embd(ctx, 0);
    int32_t hdr[3] = {n_slots, (int32_t) n_tokens, (int32_t) n_embd};
    out.write((char *) hdr, sizeof(hdr));
    out.write((char *) capture_layers.data(), (size_t) n_slots * sizeof(int32_t));

    int32_t n_toks_in = (int32_t) tokens.size();
    out.write((char *) &n_toks_in, sizeof(n_toks_in));
    std::vector<int32_t> tok_i32(tokens.begin(), tokens.end());
    out.write((char *) tok_i32.data(), tok_i32.size() * sizeof(int32_t));

    for (int slot = 0; slot < n_slots; ++slot) {
        float * data = llama_get_layer_hidden(ctx, slot);
        int64_t nt = llama_get_layer_hidden_n_tokens(ctx, slot);
        int64_t ne = llama_get_layer_hidden_n_embd(ctx, slot);
        if (!data || nt != n_tokens || ne != n_embd) {
            err = "captured hidden-state shape mismatch";
            out.close();
            std::remove(out_path.c_str());
            return false;
        }
        out.write((char *) data, nt * ne * sizeof(float));
    }

    out.close();
    n_slots_out = n_slots;
    n_tokens_out = (int32_t) n_tokens;
    n_embd_out = (int32_t) n_embd;
    return true;
}

static bool parse_request_line(
    const std::string & line,
    std::string & req_id,
    std::string & tokens_bin,
    std::string & out_bin,
    std::vector<int32_t> & capture_layers
) {
    std::stringstream ss(line);
    if (!std::getline(ss, req_id, '\t')) {
        return false;
    }
    if (!std::getline(ss, tokens_bin, '\t')) {
        return false;
    }
    if (!std::getline(ss, out_bin, '\t')) {
        return false;
    }
    std::string csv;
    if (!std::getline(ss, csv)) {
        return false;
    }
    capture_layers = parse_csv_i32(csv);
    return !req_id.empty() && !tokens_bin.empty() && !out_bin.empty() && !capture_layers.empty();
}

int main(int argc, char ** argv) {
    std::ios::sync_with_stdio(false);

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "usage: %s -m TARGET.gguf -p x -c 4096 -ngl 99\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    common_init();

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    auto cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to init ctx\n");
        llama_model_free(model);
        return 1;
    }

    std::cout << "READY\t" << params.n_ctx << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") {
            break;
        }
        std::string req_id;
        std::string tokens_bin;
        std::string out_bin;
        std::vector<int32_t> capture_layers;
        if (!parse_request_line(line, req_id, tokens_bin, out_bin, capture_layers)) {
            std::cout << "ERR\tunknown\tmalformed request" << std::endl;
            continue;
        }

        std::vector<llama_token> tokens;
        if (!read_tokens_bin(tokens_bin, tokens) || tokens.empty()) {
            std::cout << "ERR\t" << req_id << "\tcannot read tokens" << std::endl;
            continue;
        }

        llama_set_dflash_capture(ctx, capture_layers.data(), (int32_t) capture_layers.size());
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }

        llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], (llama_pos) i, {0}, true);
        }
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            std::cout << "ERR\t" << req_id << "\tdecode failed rc=" << rc << std::endl;
            continue;
        }

        int32_t n_slots = 0;
        int32_t n_tokens = 0;
        int32_t n_embd = 0;
        std::string err;
        if (!write_hidden_bin(ctx, out_bin, capture_layers, tokens, n_slots, n_tokens, n_embd, err)) {
            std::cout << "ERR\t" << req_id << "\t" << err << std::endl;
            continue;
        }

        std::cout << "OK\t" << req_id << "\t" << n_slots << "\t" << n_tokens << "\t" << n_embd << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
