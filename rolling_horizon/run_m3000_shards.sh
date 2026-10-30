#!/usr/bin/env bash
# Run the full 3000-path rolling evaluation as 6 contiguous shards, then merge.
# Shards are separate processes (COPT Threads=2, 2 workers each) because one
# Python process is GIL-bound; 6x2x2 = 24 threads on this 28-logical machine.
# Benchmark (24 t=1 LPs per process): 6x2x2 -> ~13.5 LP/s aggregate.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P="${PYTHON:-D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe}"
TOTAL="${TOTAL:-3000}"
SHARDS="${SHARDS:-6}"
SHARD_M=$((TOTAL / SHARDS))
WORKERS="${WORKERS:-2}"
THREADS="${THREADS:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
N_SCENARIOS="${N_SCENARIOS:-10}"
WINDOW="${WINDOW:-20}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-}"
TAG_PREFIX="${TAG_PREFIX:-tau_W20_M3000_shard}"
MERGED_TAG="${MERGED_TAG:-tau_W20_M3000_full_merged}"
TARGET="${TARGET:-2026-10-29 23:00:00}"
LOG="$ROOT/results/rolling_horizon/overnight_m3000_shards.log"

mkdir -p "$(dirname "$LOG")"
log() {
    echo "$(date '+%Y-%m-%dT%H:%M:%S') [shards] $*" | tee -a "$LOG"
}

cd "$ROOT"
target_epoch=$(date -d "$TARGET" +%s)
now=$(date +%s)
if [ "$now" -lt "$target_epoch" ]; then
    log "waiting until $TARGET ($((target_epoch - now))s)"
    while [ "$(date +%s)" -lt "$target_epoch" ]; do
        sleep 20
    done
fi

args_for() {
    local i="$1"
    local offset=$((i * SHARD_M))
    echo --n-scenarios "$N_SCENARIOS" --real-paths "$SHARD_M" \
        --seed-opt 2027 --path-sample first --path-offset "$offset" \
        --window-len "$WINDOW" --checkpoint-every "$CHECKPOINT_EVERY" \
        --workers "$WORKERS" --threads "$THREADS" --skip-index \
        --tag "${TAG_PREFIX}${i}" \
        ${ARCHIVE_ROOT:+--archive-root "$ARCHIVE_ROOT"}
}

latest_dir() {
    local i="$1"
    local base="${ARCHIVE_ROOT:-$ROOT}"
    ls -dt "$base"/results/rolling_horizon/${TAG_PREFIX}${i}_* \
        2>/dev/null | head -1
}

is_completed() {
    local dir="$1"
    [ -n "$dir" ] && [ -f "$dir/run.json" ] || return 1
    "$P" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8')).get('status',''))" \
        "$dir/run.json" 2>/dev/null | grep -q '^completed$'
}

log "start: shards=$SHARDS shard_m=$SHARD_M workers=$WORKERS threads=$THREADS"
for i in $(seq 0 $((SHARDS - 1))); do
    # shellcheck disable=SC2046
    "$P" -B -u rolling_horizon/run_evaluate.py $(args_for "$i") >> "$LOG" 2>&1 &
done
wait
log "first pass done"

for round in 1 2; do
    pending=0
    for i in $(seq 0 $((SHARDS - 1))); do
        dir=$(latest_dir "$i")
        if ! is_completed "$dir"; then
            pending=1
            log "resume round $round shard $i: $dir"
            # shellcheck disable=SC2046
            "$P" -B -u rolling_horizon/run_evaluate.py $(args_for "$i") \
                --resume "$dir" >> "$LOG" 2>&1 &
        fi
    done
    wait
    [ "$pending" -eq 0 ] && break
done

dirs=()
for i in $(seq 0 $((SHARDS - 1))); do
    dir=$(latest_dir "$i")
    if ! is_completed "$dir"; then
        log "shard $i not completed: $dir"
        exit 1
    fi
    # --dirs is one comma-separated argument; MSYS does not convert paths
    # inside it, so convert each shard to a native Windows path first.
    dirs+=("$(cygpath -w "$dir")")
done
joined=$(IFS=,; echo "${dirs[*]}")
log "all shards completed; merging"
"$P" -B rolling_horizon/merge_shards.py --dirs "$joined" \
    --tag "$MERGED_TAG" ${ARCHIVE_ROOT:+--archive-root "$ARCHIVE_ROOT"} \
    ${NO_SIM_NPZ:+--no-sim-npz} >> "$LOG" 2>&1
rc=$?
log "merge exit=$rc"
exit "$rc"
