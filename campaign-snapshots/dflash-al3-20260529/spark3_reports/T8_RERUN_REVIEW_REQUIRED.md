# T8 rerun result after T19 vocab-axis fix (spark-2)

Scope: driver-requested rerun from accepted T19 state. Single spark-2 only; fresh throwaway checkpoint; no production tmux touched; no DDP/FSDP/multi-host; no batch, anchor, sequence, hidden, intermediate, layer, or block-size reduction.

Locked capacity:
- max_anchors=3072
- block_size=8
- hidden_size=3072
- intermediate_size=4096
- layers=6
- total_seq_len=8192
- draft_vocab_size=32768

Result: FAILS <=2000ms/step throughput gate with loss sanity PASS.

Measured steady rows (drop first 3, n=57):
- mean total_step_ms: 2798.2 ms
- median total_step_ms: 2463.2 ms
- p90 total_step_ms: 3663.6 ms
- mean dataloader_next_ms: 0.200 ms (not data-starved in rerun)
- mean forward_ms: 563.9 ms
- mean backward_ms: 2098.5 ms
- mean optimizer_ms: 129.6 ms
- loss values logged: [10.75, 10.625, 10.0, 9.312, 8.938, 8.375]
- loss_0 sanity: PASS

Artifacts:
- /home/dnola/v18_fix/t8_reduced_vocab_rerun_result.json
- /home/dnola/v18_fix/t8_reduced_vocab_audit_latest.log
- /home/dnola/v18_fix/t8_reduced_vocab_audit_latest.audit.jsonl
- /home/dnola/v18_fix/t8_reduced_vocab_audit_latest.gpu.csv

Concrete next patch (do not bake Liger defaults yet):

Implement a custom DFlash weighted linear-CE operator for the final head, not the existing Python-token-chunked Liger path.

Patch points:
1. /home/dnola/speculators-c15-api/src/speculators/models/dflash/core.py
   - Replace the final-head path "logits = self.lm_head(self.norm(noise_embedding))" plus "compute_metrics(logits, targets, aligned_loss_mask, self.block_size)" with a fused path behind a guarded flag.
2. /home/dnola/speculators-c15-api/src/speculators/models/dflash/metrics.py
   - Split current PyTorch loss/accuracy into fallback and fused implementations.

Required semantics to validate before full trainer use:
- X = norm(noise_embedding), shape [T,H], T=24576, H=3072.
- W = lm_head.weight, shape [V,H], V=32768.
- CE per row: -log_softmax(X @ W.T)[target_i].
- DFlash loss weight: weight_i = 0 for in_block_idx=0 else exp(-max(in_block_idx-1,0)/gamma), gamma=4.0.
- Loss = sum_i CE_i * weight_i * mask_i / (sum_i weight_i * mask_i + 1e-5).
- Metrics must still produce argmax/full_acc/per-position accuracy under the same loss_mask.

Implementation shape:
- First prototype as Triton custom autograd op: dflash_weighted_linear_ce(X, W, targets, loss_mask, block_size, gamma) -> (loss, argmax_tokens, small counters).
- Stream over vocab tiles; do not materialize [24576,32768] logits/log_softmax/grad_logits.
- Forward: per-row max/argmax pass, then sum_exp/target_logit/CE pass.
- Backward: recompute tile logits or save [T] max/sum scalars; form tiled prob-minus-onehot(target) scaled by DFlash weights; accumulate dX and dW without dense grad_logits.
- First gate: tiny-shape equality vs PyTorch fallback (include masked anchor positions and per-position metrics).
- Second gate: locked-shape single-row benchmark on spark-2, then 50-step trainer audit at locked capacity.

Reason this is the concrete next patch:
- Rerun is not data-starved (dataloader mean 0.200 ms) but mean step is still 2798.2 ms.
- Backward dominates at 2098.5 ms; optimizer is only 129.6 ms.
- T18 measured the isolated required-shape PyTorch lm_head+weighted CE path at ~436.25 ms total / ~271.80 ms backward with ~6.34 GiB peak, confirming the final head remains a real target while current Liger CE is unsafe for this trainer.
- Prior T8 Liger CE on V=32768 changed early loss to 13.812 vs sane 10.75/10.625 and was slower, so it must remain disabled.
