"""Template: opt into an experimental verifier factory by name.

Use this template if you want to run the full pipeline (gen → train →
GGUF → bench) on a model family whose factory exists in
:mod:`dflash_llama.verifiers.experimental` but has NOT been validated
end-to-end by this library.

⚠️  WARNING — by definition, none of these have a known-good loss curve
or chain-pos accept rate against this library's training pipeline. The
shape metadata is plausible (sourced from public model cards) but the
DFlash drafter recipe (layer taps, drafter arch, mask token, GGUF rebake
floor, etc.) may not be correct for your target. **Always sanity-check
the loss curve and the offline eval before claiming the family works.**

If you successfully run a full pipeline against an experimental factory,
please open a PR moving the factory into the main namespace alongside a
short report (val_loss, chain-pos-1/-2 measured vs predicted, drafter
GGUF SHA, the exact tap schedule).
"""
from dflash_llama import register_verifier, load_verifier

# Import the experimental factory you want to use.
# Available factories (NOT validated):
#   kimi_k25, qwen3, qwen3_4b, qwen3_14b,
#   deepseek_v4_flash, deepseek_v4_pro,
#   nemotron3_super_120b, nemotron3_nano_30b_a3b
from dflash_llama.verifiers.experimental import kimi_k25  # ← change me

# Opt in under the canonical name.
register_verifier("kimi-k2.5", kimi_k25)

# Now load by name — and verify the shape matches your actual model.
v = load_verifier(
    "kimi-k2.5",
    hf_path="/path/to/kimi-k2.5-meta",        # ← edit
    gguf_path="/path/to/Kimi-K2.5.gguf",       # ← edit
)
print(f"loaded {v.name}: hidden={v.hidden_size}, layers={v.num_hidden_layers}, "
      f"vocab={v.vocab_size}, mask_token={v.mask_token_id}, "
      f"layer_ids={v.layer_ids}")

# From here you can use the standard library API:
#   gen = TraceGenerator(verifier=v, ...)        # generate traces
#   trainer = DFlashTrainer(traces_dir=..., verifier=v, ...)
#   trainer.smoke(...) ; trainer.train(...)
# But verify the loss curve looks sane before scaling up.
