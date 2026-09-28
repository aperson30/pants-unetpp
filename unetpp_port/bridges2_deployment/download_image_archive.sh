#!/usr/bin/env bash
# Download one immutable PanTS image shard to node-local storage. The outer
# retry is needed because wget 1.19.5 does not retry HTTP 429 responses.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: download_image_archive.sh INDEX DATA_DIR HF_REVISION" >&2
    exit 2
fi
index=$1
data_dir=$2
revision=$3
[[ "$index" =~ ^[1-9]$ ]] || { echo "ERROR: invalid shard index: $index" >&2; exit 2; }
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid dataset revision" >&2; exit 2; }
start=$(printf '%08d' $(( (index - 1) * 1000 + 1 )))
end=$(printf '%08d' $(( index * 1000 )))
archive="PanTSMini_ImageTr_${start}_${end}.tar.gz"
url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/${revision}/${archive}?download=true"
cd "$data_dir"

max_attempts=6
attempt=1
backoff=300
while :; do
    echo "image archive $index/9: download attempt $attempt/$max_attempts"
    if wget --continue --no-verbose --tries=3 --waitretry=20 \
        --timeout=120 --retry-connrefused -O "$archive" "$url"; then
        # A successful transfer must also be a valid gzip/tar archive. tar
        # checks the gzip stream before this shard is marked complete.
        tar -xzf "$archive" -C ImageTr
        rm -- "$archive"
        echo "image archive $index/9 complete"
        exit 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "ERROR: image archive $index/9 failed after $max_attempts attempts" >&2
        exit 1
    fi
    # Hugging Face documents a five-minute rate-limit window. Stagger the two
    # concurrent shards so their retry requests do not arrive together.
    wait_seconds=$((backoff + (index % 2) * 30))
    echo "image archive $index/9: waiting ${wait_seconds}s before retry; preserving partial download"
    sleep "$wait_seconds"
    if [ "$backoff" -lt 600 ]; then backoff=600; fi
    attempt=$((attempt + 1))
done
