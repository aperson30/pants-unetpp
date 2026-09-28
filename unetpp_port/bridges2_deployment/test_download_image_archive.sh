#!/usr/bin/env bash
# Offline contract test: fake wget returns two 429-style failures, then a tar.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/data/ImageTr" "$TMP/fixture/PanTS_00000001"
printf 'case fixture\n' > "$TMP/fixture/PanTS_00000001/image.txt"
printf '0\n' > "$TMP/calls"

wget() {
    local out= arg
    while [ "$#" -gt 0 ]; do
        arg=$1
        shift
        if [ "$arg" = -O ]; then out=$1; shift; fi
    done
    [[ "$arg" == *'/resolve/3b1cd61108116b58ea5c1ddb3512c1847d965f96/'* ]]
    local count
    count=$(cat "$TMP/calls")
    count=$((count + 1))
    printf '%s\n' "$count" > "$TMP/calls"
    if [ "${TEST_ALWAYS_FAIL:-0}" = 1 ] || [ "$count" -lt 3 ]; then
        if [ "$count" -eq 1 ]; then printf 'partial' > "$out"; fi
        [ -f "$out" ] || return 3
        return 8
    fi
    /usr/bin/tar -czf "$out" -C "$TMP/fixture" PanTS_00000001
}
sleep() { printf '%s\n' "$1" >> "$TMP/sleeps"; }
export -f wget sleep
export TMP

bash "$SCRIPT_DIR/download_image_archive.sh" 1 "$TMP/data" \
    3b1cd61108116b58ea5c1ddb3512c1847d965f96
[ "$(cat "$TMP/calls")" -eq 3 ]
[ "$(wc -l < "$TMP/sleeps")" -eq 2 ]
[ -f "$TMP/data/ImageTr/PanTS_00000001/image.txt" ]
[ ! -e "$TMP/data/PanTSMini_ImageTr_00000001_00001000.tar.gz" ]

printf '0\n' > "$TMP/calls"
export TEST_ALWAYS_FAIL=1
if bash "$SCRIPT_DIR/download_image_archive.sh" 1 "$TMP/data" \
    3b1cd61108116b58ea5c1ddb3512c1847d965f96; then
    echo 'ERROR: expected bounded retry failure' >&2
    exit 1
fi
[ "$(cat "$TMP/calls")" -eq 6 ]
echo 'download helper offline tests passed'
