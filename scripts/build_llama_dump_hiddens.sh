#!/usr/bin/env bash
# Build llama-dump-hiddens reproducibly from public sources.
#
# Strategy:
#   1. Clone a pinned commit of spiritbuun/buun-llama-cpp (the DFlash-enabled
#      llama.cpp fork that already has PR #22105 + the minimax-m2 cb-hook
#      patches merged). The vendored dump_hiddens.cpp / dump_hiddens_worker.cpp
#      use DFlash-specific APIs (llama_set_dflash_capture, llama_get_layer_hidden,
#      llama_get_n_layer_hiddens, etc.) that only exist in this fork — vanilla
#      ggml-org/llama.cpp does NOT have these symbols, so building against it
#      fails with 'llama_set_dflash_capture was not declared in this scope'.
#   2. Drop in our vendored examples/dump-hiddens/ source
#   3. Wire it into the cmake graph
#   4. Build with CUDA (or CPU-only via env var)
#
# Output:
#   ./build/llama.cpp-dflash/build/bin/llama-dump-hiddens
#   ./build/llama.cpp-dflash/build/bin/llama-dump-hiddens-worker (used by trace-server)
#
# Idempotent — safe to re-run; it will skip the clone if the target dir exists
# and just rebuild. To start fresh, delete build/llama.cpp-dflash/.
#
# Knobs (env vars):
#   LLAMACPP_REPO    upstream repo to clone
#                    (default: https://github.com/spiritbuun/buun-llama-cpp.git)
#   LLAMACPP_PIN     commit SHA or branch to check out
#                    (default: a known-good buun-llama-cpp master SHA)
#   BUILD_CUDA       1 (default) or 0 for CPU-only
#   JOBS             parallel build jobs (default: nproc)
#
# Usage from repo root:
#   bash scripts/build_llama_dump_hiddens.sh
#
# After success, the binary path is printed on the last line for piping
# into other scripts (e.g. `BIN=$(bash scripts/build_llama_dump_hiddens.sh | tail -1)`).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/spiritbuun/buun-llama-cpp.git}"
LLAMACPP_PIN="${LLAMACPP_PIN:-1c47881923}"
BUILD_CUDA="${BUILD_CUDA:-1}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# Auto-discover nvcc when BUILD_CUDA=1. cmake's CUDA detection requires nvcc
# on PATH or CUDACXX/CMAKE_CUDA_COMPILER set explicitly. On fresh-rebooted
# Spark hosts /usr/local/cuda/bin is often missing from PATH, causing a hard
# cmake error: "No CMAKE_CUDA_COMPILER could be found". Fix it transparently.
if [ "$BUILD_CUDA" = "1" ] && [ -z "${CUDACXX:-}" ]; then
    for _cuda_bin in /usr/local/cuda/bin/nvcc /usr/local/cuda-13/bin/nvcc \
                     /usr/local/cuda-13.0/bin/nvcc /usr/local/cuda-12/bin/nvcc \
                     /opt/cuda/bin/nvcc; do
        if [ -x "$_cuda_bin" ]; then
            export CUDACXX="$_cuda_bin"
            export PATH="$(dirname "$_cuda_bin"):$PATH"
            echo "[build] auto-discovered nvcc: $CUDACXX"
            break
        fi
    done
    if [ -z "${CUDACXX:-}" ] && ! command -v nvcc >/dev/null 2>&1; then
        echo "[build] WARNING: BUILD_CUDA=1 but nvcc not found on PATH or in" >&2
        echo "[build] /usr/local/cuda*/bin. Either install CUDA, set BUILD_CUDA=0," >&2
        echo "[build] or set CUDACXX=/path/to/nvcc explicitly." >&2
    fi
fi

VENDOR_DIR="$REPO_ROOT/vendor/dump-hiddens"
BUILD_DIR="$REPO_ROOT/build/llama.cpp-dflash"

echo "[build] upstream repo     : $LLAMACPP_REPO"
echo "[build] pinned commit/tag : $LLAMACPP_PIN"
echo "[build] cuda              : $BUILD_CUDA"
echo "[build] jobs              : $JOBS"
echo "[build] vendor_dir        : $VENDOR_DIR"
echo "[build] build_dir         : $BUILD_DIR"

# 1. clone (if needed)
if [ ! -d "$BUILD_DIR/.git" ]; then
    echo "[build] cloning $LLAMACPP_REPO and checking out $LLAMACPP_PIN"
    mkdir -p "$(dirname "$BUILD_DIR")"
    # Try shallow-by-branch first (works for tags and branch names); on
    # failure, fall back to full clone + checkout (works for arbitrary
    # commit SHAs).
    if ! git clone --depth 1 --branch "$LLAMACPP_PIN" \
            "$LLAMACPP_REPO" "$BUILD_DIR" 2>/dev/null; then
        echo "[build] shallow-branch clone failed; falling back to full clone"
        git clone "$LLAMACPP_REPO" "$BUILD_DIR"
        ( cd "$BUILD_DIR" && git checkout "$LLAMACPP_PIN" )
    fi
fi

cd "$BUILD_DIR"

# 2. drop in our vendored example (overwrite any prior copy)
mkdir -p examples/dump-hiddens
cp -f "$VENDOR_DIR/dump_hiddens.cpp"        examples/dump-hiddens/
cp -f "$VENDOR_DIR/dump_hiddens_batch.cpp"  examples/dump-hiddens/
cp -f "$VENDOR_DIR/dump_hiddens_worker.cpp" examples/dump-hiddens/
cp -f "$VENDOR_DIR/CMakeLists.txt"          examples/dump-hiddens/

# 3. register the example in examples/CMakeLists.txt (idempotent)
if ! grep -q "add_subdirectory(dump-hiddens)" examples/CMakeLists.txt; then
    echo "[build] registering dump-hiddens in examples/CMakeLists.txt"
    echo "" >> examples/CMakeLists.txt
    echo "add_subdirectory(dump-hiddens)" >> examples/CMakeLists.txt
fi

# 4. build
mkdir -p build
cd build

CMAKE_FLAGS=()
if [ "$BUILD_CUDA" = "1" ]; then
    CMAKE_FLAGS+=("-DGGML_CUDA=ON")
fi
CMAKE_FLAGS+=("-DCMAKE_BUILD_TYPE=Release")
# build only what we need — saves several minutes
CMAKE_FLAGS+=("-DLLAMA_BUILD_TESTS=OFF" "-DLLAMA_BUILD_SERVER=OFF")

echo "[build] cmake ${CMAKE_FLAGS[*]} .."
cmake "${CMAKE_FLAGS[@]}" ..

echo "[build] make -j$JOBS llama-dump-hiddens llama-dump-hiddens-batch llama-dump-hiddens-worker"
cmake --build . -j "$JOBS" --target llama-dump-hiddens llama-dump-hiddens-batch llama-dump-hiddens-worker

BIN_PATH="$BUILD_DIR/build/bin/llama-dump-hiddens"
WORKER_PATH="$BUILD_DIR/build/bin/llama-dump-hiddens-worker"
if [ ! -x "$BIN_PATH" ]; then
    echo "[build] ERROR: $BIN_PATH not built" >&2
    exit 1
fi
if [ ! -x "$WORKER_PATH" ]; then
    echo "[build] ERROR: $WORKER_PATH not built" >&2
    exit 1
fi

echo "[build] OK"
echo "[build] binary (one-shot)  : $BIN_PATH"
echo "[build] binary (server worker): $WORKER_PATH"
echo "$BIN_PATH"
