#!/usr/bin/env bash
set -euo pipefail
cd "/home/<USER>/lucebox-latest-20260603/server"
export CUDA_VISIBLE_DEVICES=0
export DFLASH_PERPOS_PROBE=1
exec stdbuf -oL -eL ./build-sm121/dflash_server "/home/<USER>/models/MiniMax-M2.7-GGUF/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf" --draft "/home/<USER>/t_5567a42d/golden_step_00020000/model_lucebox_layout.safetensors" --port 8097 --host 127.0.0.1 --max-ctx 2048 --default-max-tokens 128 --hard-limit-reply-budget 0 --model-name minimax-luce-golden
