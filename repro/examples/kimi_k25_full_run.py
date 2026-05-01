"""
Kimi-K2.5 example — proves the verifier abstraction works for non-MiniMax models.

The library's ``verifiers/`` registry is the model-family abstraction. Each
verifier carries the model-specific knowledge the trace generator and trainer
need:

- ``hidden_size`` — must match the trainer's ``--hidden-size``
- ``vocab_size`` and ``mask_token_id`` — must match the verifier's tokenizer
- ``num_hidden_layers`` and ``layer_ids`` — must match what the verifier emits
- ``drafter_arch`` — what arch the drafter should use (typically ``qwen3``
  for everything since it's the canonical small-transformer flavor)

Switching from MiniMax-M2.7 to Kimi-K2.5 is **a single config change**.

The DFlash paper's published drafter for Kimi-K2.5 uses these layer taps:
    [1, 12, 24, 35, 47, 58]
Versus MiniMax-M2.7's:
    [2, 16, 30, 45, 59, 61]

Both are six evenly-spaced taps from the verifier's hidden states. The library
already encodes them in the registered verifiers.

Prerequisites
-------------
    pip install -e /path/to/dflash-llama
    # plus a Kimi-K2.5 GGUF (e.g. Kimi-K2.5-IQ4_XS-00001-of-N.gguf)
    # plus the Kimi tokenizer (vocab=163840, mask=163838) reachable as an HF dir

Edit the paths and run::

    python kimi_k25_full_run.py
"""
from pathlib import Path

from dflash_llama import DFlashTrainer, TraceGenerator, load_verifier

# --- edit for your environment ---
GGUF       = "/path/to/Kimi-K2.5-IQ4_XS-00001-of-N.gguf"
LLAMA_BIN  = "/path/to/buun-llama-cpp/build/bin/llama-dump-hiddens"
PROMPTS    = "/path/to/prompts_arrow_dir"
HF_VERIFIER = "/path/to/Kimi-K2.5"   # for the trainer's tokenizer + config
WORK       = Path("/path/to/runs/kimi_k25_full")
# ----------------------------------

verifier = load_verifier("kimi-k2.5", gguf_path=GGUF, hf_path=HF_VERIFIER)
print(f"verifier {verifier.name}: hidden={verifier.hidden_size} "
      f"layer_ids={verifier.layer_ids}  vocab={verifier.vocab_size}  "
      f"mask_token_id={verifier.mask_token_id}")

# Generate 6,500 traces — note this is identical to the MiniMax example,
# only the verifier differs
gen = TraceGenerator(
    verifier=verifier,
    storage="fp8_per_tensor_scale",
    backend="llamacpp_gguf",
    backend_kwargs={"binary": LLAMA_BIN},
)
gen.generate(
    prompts=PROMPTS,
    output_dir=str(WORK / "traces"),
    rows=range(0, 6500),
    state_path=str(WORK / "state.json"),
    max_seq_len=2048,
)

trainer = DFlashTrainer(
    traces_dir=str(WORK / "traces"),
    verifier=verifier,
    num_layers=5,
    draft_vocab_size=32768,
    paired_dir=str(WORK / "paired"),
)
trainer.prepare()
trainer.smoke(timeout_sec=90, save_to=str(WORK / "smoke_ckpt"))
trainer.train(
    save_to=str(WORK / "checkpoints"),
    epochs=17, lr=3e-5, max_anchors=512,
)
trainer.offline_eval(checkpoint=str(WORK / "checkpoints" / "checkpoint_best"))
print(f"Done. Trained Kimi-K2.5 DFlash drafter at {WORK / 'checkpoints'}")
