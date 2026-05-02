"""Example: convert a trained DFlash drafter to a buun-loadable GGUF.

Run on spark-1 (or any machine with buun-llama-cpp + the vllm venv installed)::

    python repro/examples/03_export_gguf.py
"""
from dflash_llama import export_to_gguf
from dflash_llama.inference import verify_gguf_metadata

# 1. Convert: ckpt → GGUF
gguf_path = export_to_gguf(
    checkpoint="/home/user/bf16only_for_gguf",                # speculators-format ckpt dir
    output_path="/home/user/models/MiniMax-M2.7-DFlash-bf16only.gguf",
    verifier_meta_dir="/home/user/iq4_full_run/verifier_meta",  # tokenizer source
    buun_repo="/home/user/buun-llama-cpp",                    # buun checkout
    # venv_python autodetects /home/user/venvs/vllm/bin/python3
    # rebake_floor=-65504.0 (the d2t bug fix — don't change without reading the skill)
)

# 2. Verify metadata sanity (architecture, target_layer_ids, block_size, ...)
meta = verify_gguf_metadata(gguf_path)
print(f"GGUF: {gguf_path}")
print(f"  arch:               {meta['general.architecture']}")        # must be 'dflash-draft'
print(f"  target_layer_ids:   {meta['dflash-draft.dflash.target_layer_ids']}")
print(f"  block_size:         {meta['dflash-draft.dflash.block_size']}")
print(f"  mask_token_id:      {meta['dflash-draft.dflash.mask_token_id']}")
print(f"  total_tensors:      {meta['total_tensors']}")
print(f"  size:               {meta['file_size_bytes']/1e9:.2f} GB")
