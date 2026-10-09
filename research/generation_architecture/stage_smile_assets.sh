#!/usr/bin/env bash
# Model artifacts only: no datasets, optimizer states, GPU jobs or shared-env edits.
set -euo pipefail
umask 077
root=/u/asanjeev/generation_architecture_20261008_v1
hf=/u/asanjeev/.local/bin/hf
mkdir -p "$root/assets"
printf 'Started UTC: %s\n' "$(date -u +%FT%TZ)"
"$hf" download MitakaKuma/SMILE \
  SMILE/unet/config.json SMILE/unet/diffusion_pytorch_model.safetensors \
  autoencoder/vae/config.json autoencoder/vae/diffusion_pytorch_model.safetensors \
  --revision eb792bdfa9f90a553a2ffb0bd52441b81a0bb8b0 \
  --local-dir "$root/assets/smile"
"$hf" download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  model_index.json scheduler/scheduler_config.json \
  tokenizer/merges.txt tokenizer/special_tokens_map.json \
  tokenizer/tokenizer_config.json tokenizer/vocab.json \
  text_encoder/config.json text_encoder/model.safetensors \
  unet/config.json unet/diffusion_pytorch_model.safetensors \
  --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
  --local-dir "$root/assets/sd15"
find "$root/assets" -type f \( -name '*.safetensors' -o -name '*.json' -o -name '*.txt' \) \
  -not -path '*/.cache/*' -print0 | sort -z | xargs -0 sha256sum > "$root/assets/SHA256SUMS"
printf 'Assets downloaded and hashed UTC: %s\n' "$(date -u +%FT%TZ)"
# This marker certifies download/hash completion, not inference or model quality.
touch "$root/assets/DOWNLOAD_COMPLETE"
