#!/bin/bash
# Rip non-genomics, spark-1-unique data → spark-6 (over QSFP), verify, then delete from spark-1.
#
# Per user directive: "Don't delete anything we don't have duplicated, instead rip it on spark-6."
# Strict reading: we rip duplicates too, just to have a unified archive on spark-6.
# Order: small unique items first (proves the route works on cheap data), then bulk dups.
#
# Verification: each rsync's destination is checked for byte-size + file-count match before
# the source is deleted. Aborts on mismatch — no data destroyed without verified replica.
#
# Spark-1 → Spark-6 over QSFP: 192.168.200.6, sustained ~500 MB/s on this fabric.
#
# RUN ON: spark-1 only. Detached via tmux for kill-safety.

set -u
LOG=/home/user/spark1_cleanup.log
DEST_HOST=192.168.200.6
DEST_BASE=/home/user/spark1_archive

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

rip() {
    local src="$1"
    local sub="$2"
    local dst="${DEST_BASE}/${sub}"
    log "=== RIP: $src → spark-6:$dst ==="
    if [ ! -e "$src" ]; then log "  SKIP: src missing"; return; fi
    local src_size src_count
    src_size=$(du -sb "$src" 2>/dev/null | cut -f1)
    src_count=$(find "$src" -type f 2>/dev/null | wc -l)
    log "  src: ${src_size} bytes, ${src_count} files"

    ssh ${DEST_HOST} "mkdir -p $(dirname $dst)"
    rsync -aH --info=progress2 --no-compress "$src" "${DEST_HOST}:${dst}" 2>&1 | tail -5 | tee -a "$LOG"
    local rsync_rc=${PIPESTATUS[0]}
    if [ "$rsync_rc" -ne 0 ]; then
        log "  ABORT: rsync exit=$rsync_rc — NOT deleting source"
        return 1
    fi

    local dst_size dst_count
    dst_size=$(ssh ${DEST_HOST} "du -sb '$dst' 2>/dev/null | cut -f1")
    dst_count=$(ssh ${DEST_HOST} "find '$dst' -type f 2>/dev/null | wc -l")
    log "  dst: ${dst_size} bytes, ${dst_count} files"
    if [ "$src_size" != "$dst_size" ] || [ "$src_count" != "$dst_count" ]; then
        log "  ABORT: size/count mismatch — NOT deleting source"
        return 1
    fi
    log "  VERIFIED. Deleting source."
    rm -rf "$src" && log "  DELETED $src" || log "  DELETE FAILED $src"
}

log "########### START spark-1 cleanup → spark-6 (QSFP) ###########"
df -h /home | tail -1 | tee -a "$LOG"

# UNIQUE items first (small, prove the route)
rip /home/user/full_epoch5_for_gguf            full_epoch5_for_gguf
rip /home/user/full_epoch5_prepped             full_epoch5_prepped
rip /home/user/full_final_for_gguf             full_final_for_gguf
rip /home/user/full_final_prepped              full_final_prepped
rip /home/user/models/MiniMax-M2.7-DFlash-FULL-epoch5.gguf  models/MiniMax-M2.7-DFlash-FULL-epoch5.gguf
rip /home/user/models/MiniMax-M2.7-DFlash-FULL-final.gguf   models/MiniMax-M2.7-DFlash-FULL-final.gguf
rip /home/user/dflash_minimax/checkpoints      dflash_minimax_checkpoints

# True dup (md5-verified identical to live)
log "=== DELETE (md5-identical dup): MiniMax-M2.7-DFlash.gguf.bak ==="
rm -f /home/user/models/MiniMax-M2.7-DFlash.gguf.bak && log "  deleted"

# Bulk DUP items last (big — these are the disk wins)
rip /home/user/dflash_minimax/data/preprocessed_5L_FP8/hs_staging   dflash_minimax/data/preprocessed_5L_FP8/hs_staging
rip /home/user/models/MiniMax-M2.7-FP8                              models/MiniMax-M2.7-FP8

log "########### DONE ###########"
df -h /home | tail -1 | tee -a "$LOG"
log "Free space delta visible above."
