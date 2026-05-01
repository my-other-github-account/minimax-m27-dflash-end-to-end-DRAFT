# Vendored: `examples/dump-hiddens` for llama.cpp

This is the minimal `llama-dump-hiddens` example needed to capture verifier hidden states for DFlash trace generation. It is **vendored into this repo** so the build is fully reproducible from public sources (just `ggml-org/llama.cpp` upstream + this one example).

## Provenance

These files originated in [`spiritbuun/buun-llama-cpp`](https://github.com/spiritbuun/buun-llama-cpp), an experimental fork. The example itself is small (~140 lines C++) and depends only on stable llama.cpp public APIs (`llama.h`, `common.h`, `arg.h`). To keep this repo self-contained we vendor the source here and build it on top of a pinned upstream `ggml-org/llama.cpp` commit.

## Files

| file | purpose |
|---|---|
| `dump_hiddens.cpp` | single-prompt CLI, writes hidden states to binary blob |
| `dump_hiddens_batch.cpp` | batched CLI, used in batch trace generation |
| `CMakeLists.txt` | drop-in cmake target definition |

## Building

Don't build by hand — use [`scripts/build_llama_dump_hiddens.sh`](../../scripts/build_llama_dump_hiddens.sh) at the repo root. It clones the pinned upstream commit, drops these files in, and produces `bin/llama-dump-hiddens`.

## Output binary format

Little-endian:

```
int32  n_layers
int32  n_tokens
int32  n_embd
int32[n_layers]  capture_layer_ids
int32  n_tokens_in
int32[n_tokens_in] token_ids
float32[n_layers, n_tokens, n_embd]   row-major
```

This is consumed by `dflash_llama.generation.backends.llamacpp_gguf.LlamaCppGGUFBackend`.
