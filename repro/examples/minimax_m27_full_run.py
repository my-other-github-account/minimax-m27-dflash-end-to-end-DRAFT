"""End-to-end MiniMax-M2.7 trace generation + DFlash drafter training.

This is the canonical 30-line walkthrough. It assumes you have:

  - a GGUF MiniMax-M2.7-UD-IQ4_XS at $GGUF
  - the bf16 HF model at $HF (for --verifier-name-or-path during training)
  - an HF prompts arrow dataset at $PROMPTS with input_ids + loss_mask cols
  - llama-dump-hiddens on $PATH (or override via backend_kwargs={"binary": ...})

Generation is resumable (re-running picks up where it left off via state.json).
"""
from dflash_llama import DFlashTrainer, TraceGenerator, load_verifier

GGUF = "/path/to/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
HF = "/home/user/models/MiniMax-M2.7-FP8"
PROMPTS = "/path/to/prompts_arrow_dir"
TRACES = "/path/to/traces"
PAIRED = "/path/to/paired"
CKPT = "/path/to/checkpoint"


def main():
    verifier = load_verifier("minimax-m2.7-iq4-xs", gguf_path=GGUF, hf_path=HF)

    # 1) Generate self-describing fp8 traces (resumable)
    gen = TraceGenerator(verifier=verifier, storage="fp8_per_tensor_scale")
    gen.generate(prompts=PROMPTS, output_dir=TRACES,
                 rows=range(0, 1000), max_seq_len=2048)

    # 2) Train the drafter
    trainer = DFlashTrainer(
        traces_dir=TRACES, verifier=verifier,
        num_layers=5, draft_vocab_size=32768,
        paired_dir=PAIRED,
    )
    trainer.prepare()
    smoke = trainer.smoke(timeout_sec=90)
    assert smoke.passed, smoke.message
    trainer.train(save_to=CKPT, epochs=17, lr=3e-5, max_anchors=512)

    # 3) Validate
    metrics = trainer.offline_eval(
        checkpoint=f"{CKPT}/checkpoint_best", max_batches=60,
    )
    print("val metrics:", metrics)


if __name__ == "__main__":
    main()
