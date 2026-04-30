# Pool completeness debugging — what to do if the audit fails

## What the audit checks

`repro/scripts/training/audit_pool_completeness.py` hashes the `token_ids` field of every safetensor file in your canonical pool, and the same field for every safetensor on a peer rank's `hs_staging/`. It then compares the two sets:

- **Ideal:** `pool ⊆ peer-staging`, intersection equals pool size, zero samples in staging that aren't in pool.
- **Failure:** non-zero count of "samples in peer-staging not in pool" → the validator dropped real data, OR the validator didn't get to scan some staging files before they were rotated/cleaned, OR there was a crash mid-promotion.

## Reference: what the production cluster looked like

For the production §1 run:

```
Local  pool  (canonical, validator output):     6,515 files,  6,515 unique by token_ids
Peer   staging  (raw datagen output):          17,657 files,  6,515 unique by token_ids
intersection (sha256 of token_ids):             6,515
in peer-staging not in pool:                        0   ← lossless dedup ✓
in pool not in peer-staging:                        0
```

The 17,657 vs 6,515 ratio is from offline-datagen retries: the same logical sample gets written multiple times (different `cmpl-<request_id>` filenames, identical `token_ids`). The validator dedupes via first-4KB sha when promoting to pool, so each unique sample collapses to one `hs_<N>.safetensors`. If your numbers look similar, you're fine.

## When the audit reports "in peer-staging not in pool: > 0"

### Diagnosis order

1. **Did the validator daemon crash partway through the run?** Check its log:
   ```bash
   tail -100 ${WORKSPACE}/dflash_minimax/logs/validator-5L.log
   ```
   Look for tracebacks, `Killed`, or large gaps between successive `[validator]` log entries. If the validator died at hour 4 of an 8-hour generation, everything written after is orphaned in `hs_staging/`.

2. **Are the missing files actually valid hidden states?** Pull one of the unmatched filenames and inspect:
   ```python
   from safetensors.torch import safe_open
   with safe_open("/path/on/peer/cmpl-XXX-0-YYY.safetensors", framework="pt") as f:
       print(f.keys())
       hs = f.get_tensor("hidden_states")
       print(hs.shape, hs.dtype)
       print("any NaN:", hs.isnan().any().item())
       print("any Inf:", hs.isinf().any().item())
   ```
   If they have NaN/Inf or wrong shape, they were correctly rejected by the validator. If they look clean, the validator just never saw them.

3. **Were they quarantined on the canonical node, not pool?** Quarantine is the validator's "looks-bad" bucket; the audit script only compares `pool` and peer staging. A genuinely missing-from-pool sample might still be in `hs_quarantine/`. **Do not touch quarantine to recover them** — quarantine entries failed validation for a reason. If you suspect the validator was over-aggressive, fix the validator's thresholds (R62 zero-rate gate, NaN check) and rerun §1.

### Recovery: re-run the validator on the live staging

If the validator simply died early and the missing files are clean, you can restart the validator daemon and have it walk the existing staging:

```bash
ssh canonical-node "systemd-run --user --unit=validator-5L --collect \
  bash ${WORKSPACE}/repos/speculators-repro/repro/scripts/generation/validator_daemon_5L.py"
```

The daemon's monotonic numbering picks up from `state['pool_idx']` so newly promoted files extend the pool without overwriting. Re-run the audit afterwards to confirm.

### Recovery: rebuild the pool from peer staging directly

If the canonical node's staging is also gone (cleaned up, disk full, etc.) but a peer rank's staging still has the data, you can rebuild the pool there:

```bash
ssh peer-node "systemd-run --user --unit=validator-rebuild --collect \
  bash -c 'cd ${WORKSPACE}/dflash_minimax/data/preprocessed_5L_FP8 && \
  cp -r ${WORKSPACE}/repos/speculators-repro/repro/scripts/generation/validator_daemon_5L.py /tmp/ && \
  STAGING=hs_staging POOL=hs_clean_pool python /tmp/validator_daemon_5L.py'"
```

The validator's logic is content-deduplicating (first-4KB sha) so it doesn't matter which rank's staging you point it at — output is identical.

## When the audit reports "in pool not in peer-staging: > 0"

This is rare but possible, and concerning. It means your pool contains samples that the peer rank you queried never produced. Possible causes:

- **You're auditing the wrong peer.** Different peer ranks may have processed different prompt-source rotations during the loop. Try another peer.
- **Staging was rotated/deleted on the peer.** If the peer ran low on disk and someone cleaned `hs_staging/`, then-yes the peer no longer holds those originals. The pool is still valid (those samples were verified-and-promoted at the time); the missing-from-staging is a peer-side disk state, not a data-integrity issue.
- **Pool was hand-edited.** Did anyone manually copy files into `hs_clean_pool/` outside the validator? Check timestamps and `validator_state.json`'s recorded `pool_idx` vs actual file count.

Audit is a one-way check. `pool ⊄ peer-staging` ≠ `pool is corrupt`. The real integrity test is whether each pool file's `token_ids` matches a known prompt source — that's what `build_paired_dataset.py` does, and a 100% match rate there is the strongest signal.

## When to stop debugging and just train on what you have

If your pool has ≥ 95% of what you expected and the missing samples don't seem systematically biased (e.g. not all from one prompt source), training on the slightly-smaller paired set is fine. The §2.7 reference numbers were achieved on 6,515 files; substantially fewer (say, 4,000+) still produces a working drafter, just with somewhat lower per-position accuracy.

If you're missing >20% or the missing samples ARE systematically biased (one source is barely represented), regenerate that source — re-run §1 with the appropriate `prompt_loader` invocation pointing at the under-covered source.
