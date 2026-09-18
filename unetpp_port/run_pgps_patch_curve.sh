#!/usr/bin/env bash
set -uo pipefail

# Run every point in its own process so CUDA/driver state cannot accumulate across patch sizes.
# This script intentionally benchmarks eager AMP: it isolates the PGPS patch-size effect from the
# separately measured torch.compile effect.

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/pants_unetpp/bin/python}"
RESULT_DIR="${RESULT_DIR:-$HOME/pgps_patch_curve_results}"
MIN_AVAILABLE_KIB=$((70 * 1024 * 1024))
PATCHES=("64,128,160" "64,128,192" "64,160,192" "64,160,224")
ARCHES=("unetpp" "plainunet")

mkdir -p "$RESULT_DIR"

available_kib() {
    awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

echo "$(date --iso-8601=seconds) PGPS_CURVE_START host=$(hostname)"
echo "python=$PYTHON_BIN result_dir=$RESULT_DIR min_available_kib=$MIN_AVAILABLE_KIB"

for arch in "${ARCHES[@]}"; do
    for patch in "${PATCHES[@]}"; do
        before_kib=$(available_kib)
        echo "$(date --iso-8601=seconds) START arch=$arch patch=$patch mem_available_kib=$before_kib"
        if (( before_kib < MIN_AVAILABLE_KIB )); then
            echo "$(date --iso-8601=seconds) ABORT insufficient host-memory headroom"
            exit 2
        fi

        output="$RESULT_DIR/${arch}_patch_${patch//,/_}.json"
        if timeout --signal=TERM --kill-after=30s 15m \
            "$PYTHON_BIN" pgps_patch_curve_bench.py \
            --arch "$arch" --patch "$patch" --output "$output"; then
            echo "$(date --iso-8601=seconds) FINISH arch=$arch patch=$patch"
        else
            status=$?
            echo "$(date --iso-8601=seconds) FAIL arch=$arch patch=$patch status=$status"
            exit "$status"
        fi

        # A process exit should return nearly all of its CUDA/UMA footprint. Refuse to continue if
        # host memory does not recover within one minute.
        recovered=0
        for _ in $(seq 1 12); do
            after_kib=$(available_kib)
            if (( after_kib >= MIN_AVAILABLE_KIB )); then
                recovered=1
                break
            fi
            sleep 5
        done
        echo "$(date --iso-8601=seconds) RECOVERY arch=$arch patch=$patch mem_available_kib=$after_kib"
        if (( recovered == 0 )); then
            echo "$(date --iso-8601=seconds) ABORT memory did not recover after child exit"
            exit 3
        fi
    done
done

echo "$(date --iso-8601=seconds) PGPS_CURVE_DONE"
