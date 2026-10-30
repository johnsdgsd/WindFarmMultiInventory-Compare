#!/usr/bin/env bash
# Resume the four M3000 shards that stalled at t=49/51 on 2026-10-30.
# Runs sequentially to avoid the multi-process COPT stall; each LP trial is
# capped at 120 s, and each shard process at 1 h.  Never restarts from t=1.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P="${PYTHON:-D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe}"
DATE_TAG="${DATE_TAG:-20261029_212144}"
LOG="$ROOT/results/rolling_horizon/resume_m3000_20261030.log"
mkdir -p "$(dirname "$LOG")"
log() {
    echo "$(date '+%Y-%m-%dT%H:%M:%S') [resume] $*" | tee -a "$LOG"
}
cd "$ROOT"

shard_dir() {
    echo "$ROOT/results/rolling_horizon/tau_W20_M3000_shard${1}_${DATE_TAG}"
}

is_completed() {
    "$P" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8')).get('status',''))" \
        "$(cygpath -w "$1/run.json")" 2>/dev/null | grep -q '^completed$'
}

log "resume stalled shards sequentially; time_limit=120s per COPT trial"
for i in 0 2 3 4; do
    d=$(shard_dir "$i")
    if [ ! -f "$d/checkpoint.npz" ]; then
        log "shard $i: no checkpoint, abort"
        exit 1
    fi
    log "shard $i: resume from $(python -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('t'))" "$(cygpath -w "$d/checkpoint.json")")"
    timeout 3600 "$P" -B -u rolling_horizon/run_evaluate.py \
        --n-scenarios 10 --real-paths 500 --seed-opt 2027 \
        --path-sample first --path-offset "$((i * 500))" \
        --window-len 20 --checkpoint-every 1 \
        --workers 2 --threads 2 --skip-index --time-limit 120 \
        --tag "tau_W20_M3000_shard${i}" \
        --resume "$(cygpath -w "$d")" \
        >> "/tmp/resume_m3000_shard${i}.log" 2>&1
    rc=$?
    if is_completed "$d"; then
        log "shard $i: exit=$rc completed=yes"
    else
        log "shard $i: exit=$rc completed=no; stopping before merge"
        exit 1
    fi
done

log "all four resumed shards completed; merging all six"
dirs=()
for i in 0 1 2 3 4 5; do
    d=$(shard_dir "$i")
    if ! is_completed "$d"; then
        log "shard $i not completed; abort merge"
        exit 1
    fi
    dirs+=("$(cygpath -w "$d")")
done
joined=$(IFS=,; echo "${dirs[*]}")
"$P" -B rolling_horizon/merge_shards.py --dirs "$joined" \
    --tag tau_W20_M3000_full_merged >> "$LOG" 2>&1
rc=$?
log "merge exit=$rc"
exit "$rc"
