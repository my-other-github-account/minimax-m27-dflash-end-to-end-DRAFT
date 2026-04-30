#!/usr/bin/env python3
"""
Continuous validator: scans hs_staging/, validates each file (no NaN/Inf, has all 4 layers),
moves clean -> hs_clean_pool with monotonic numbering, moves bad -> hs_quarantine.
Runs forever; sleeps 30s between scans. Logs progress.
"""
import os, sys, shutil, hashlib, time, json
from safetensors.torch import load_file

STAGING = '${DATA_ROOT}/preprocessed_5L_FP8/hs_staging'
POOL = '${DATA_ROOT}/preprocessed_5L_FP8/hs_clean_pool'
QUAR = '${DATA_ROOT}/preprocessed_5L_FP8/hs_quarantine'
STATE = '${DATA_ROOT}/preprocessed_5L_FP8/validator_state.json'
LOG = '${WORKSPACE}/logs/validator-5L.log'

os.makedirs(POOL, exist_ok=True)
os.makedirs(QUAR, exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

# Load state
state = {'pool_idx': 0, 'pool_hashes': []}
if os.path.exists(STATE):
    with open(STATE) as f:
        state = json.load(f)
        state['pool_hashes'] = set(state.get('pool_hashes', []))
else:
    # Initialize from existing pool
    existing = [f for f in os.listdir(POOL) if f.startswith('hs_') and f.endswith('.safetensors')]
    state['pool_idx'] = max([int(f.replace('hs_', '').replace('.safetensors', '')) for f in existing], default=-1) + 1
    state['pool_hashes'] = set()
    for f in existing:
        try:
            with open(os.path.join(POOL, f), 'rb') as fh:
                state['pool_hashes'].add(hashlib.sha256(fh.read(4096)).hexdigest())
        except: pass

log(f"Validator started. Pool starts at idx={state['pool_idx']}, {len(state['pool_hashes'])} known hashes")

def save_state():
    with open(STATE, 'w') as f:
        s = dict(state)
        s['pool_hashes'] = list(state['pool_hashes'])
        json.dump(s, f)

def validate_one(path):
    """Return 'clean', 'nan', or 'error'."""
    try:
        d = load_file(path)
        # Must have hidden_states key
        if 'hidden_states' not in d:
            return 'error', 'no hidden_states'
        for k, v in d.items():
            if v.dtype.is_floating_point:
                if v.isnan().any().item():
                    return 'nan', f'{k} has NaN'
                if v.isinf().any().item():
                    return 'nan', f'{k} has Inf'
        # Sanity check shape: hidden_states should be 3D [seq, n_layers, hidden] or [n_layers, seq, hidden]
        hs = d['hidden_states']
        if hs.dim() != 3 or hs.shape[0] < 3:
            return 'error', f'shape {hs.shape}'
        # R62_ZERO_RATE_GATE: per-layer zero-rate gate (catches R33 silent zero-poisoning at deep layers).
        # If any layer has >30% all-zero rows, quarantine. Real activations don't go all-zero;
        # all-zero comes from R33 NaN-fallback masking bf16 overflow.
        # Layout in 5L extraction: hs is [seq, n_layers, hidden] where n_layers = aux_count + 1 (last_hidden).
        if hs.shape[1] < hs.shape[0]:  # axis-1 is layer dim (typical: seq >> n_layers)
            for L in range(hs.shape[1]):
                n_zero = (hs[:, L, :].float().abs().sum(dim=-1) == 0).sum().item()
                zero_rate = n_zero / hs.shape[0]
                if zero_rate > 0.30:
                    return 'nan', f'L{L} zero-rate {zero_rate:.0%} (R33 overflow at deep layer)'
        else:  # axis-0 is layer dim (legacy or rotated layout)
            for L in range(hs.shape[0]):
                n_zero = (hs[L].float().abs().sum(dim=-1) == 0).sum().item()
                zero_rate = n_zero / hs.shape[1]
                if zero_rate > 0.30:
                    return 'nan', f'L{L} zero-rate {zero_rate:.0%} (R33 overflow at deep layer)'
        # R64_EXTENDED_VALIDATION: zero-collapse, constant-collapse, pure-echo guards.
        # Detects three pathological generation modes that R62 doesn't catch.
        # Layout: hs is [seq, n_layers, hidden] when shape[1] < shape[0] (typical),
        # token_ids is [seq] int64. n_layers = 6 (5 aux + 1 verifier_last_hidden_states at index 5).
        try:
            tok = d.get('token_ids', None)
            # Check 5: pure-echo proxy — sequence too short to have meaningful generation
            if tok is not None and tok.numel() < 10:
                return 'nan', f'R64 pure_echo seq_len={tok.numel()}'

            # Determine layer axis (same logic as R62)
            if hs.shape[1] < hs.shape[0]:
                # [seq, n_layers, hidden]
                last_layer = hs[:, -1, :].float()  # (seq, hidden) — verifier_last_hidden_states
            else:
                # [n_layers, seq, hidden]
                last_layer = hs[-1].float()  # (seq, hidden)

            # Check 3: zero_collapse — final layer tap near-zero everywhere
            mean_abs = last_layer.abs().mean().item()
            if mean_abs < 0.01:
                return 'nan', f'R64 zero_collapse final-tap mean_abs={mean_abs:.5f}'

            # Check 4: constant_collapse — token-to-token L2 distance variance too low
            if last_layer.shape[0] >= 4:
                diffs = (last_layer[1:] - last_layer[:-1]).norm(dim=-1)  # (seq-1,)
                var = diffs.var().item()
                if var < 1e-6:
                    return 'nan', f'R64 constant_collapse L2-diff var={var:.2e}'
        except Exception as e:
            return 'error', f'R64 check raised: {str(e)[:80]}'
        # END R64_EXTENDED_VALIDATION

        return 'clean', 'ok'
    except Exception as e:
        return 'error', str(e)[:80]

scan_count = 0
total_clean = 0
total_nan = 0
total_err = 0

while True:
    scan_count += 1
    files = sorted([f for f in os.listdir(STAGING) if f.startswith('hs_') and f.endswith('.safetensors')])
    if not files:
        if scan_count % 4 == 0:
            log(f"scan #{scan_count}: staging empty, sleeping...")
        time.sleep(30)
        continue

    batch_clean = batch_nan = batch_err = batch_dup = 0
    for f in files:
        p = os.path.join(STAGING, f)
        # Wait if file is still being written (mtime within last 3s)
        try:
            if time.time() - os.path.getmtime(p) < 3:
                continue
        except FileNotFoundError:
            continue

        # Dedupe
        try:
            with open(p, 'rb') as fh:
                h = hashlib.sha256(fh.read(4096)).hexdigest()
        except: 
            continue
        if h in state['pool_hashes']:
            try: os.remove(p)
            except: pass
            batch_dup += 1
            continue

        verdict, reason = validate_one(p)
        if verdict == 'clean':
            new_name = f"hs_{state['pool_idx']}.safetensors"
            try:
                shutil.move(p, os.path.join(POOL, new_name))
                state['pool_hashes'].add(h)
                state['pool_idx'] += 1
                batch_clean += 1
            except Exception as e:
                log(f"  move-fail {f}: {e}")
                batch_err += 1
        elif verdict == 'nan':
            try: shutil.move(p, os.path.join(QUAR, f))
            except: pass
            batch_nan += 1
        else:  # error
            try: shutil.move(p, os.path.join(QUAR, f"err_{f}"))
            except: pass
            batch_err += 1

    total_clean += batch_clean
    total_nan += batch_nan
    total_err += batch_err
    if batch_clean + batch_nan + batch_err + batch_dup > 0:
        log(f"scan #{scan_count}: clean+={batch_clean} nan={batch_nan} err={batch_err} dup={batch_dup} | pool_size={state['pool_idx']} totals: clean={total_clean} nan={total_nan} err={total_err}")
    save_state()
    time.sleep(30)
