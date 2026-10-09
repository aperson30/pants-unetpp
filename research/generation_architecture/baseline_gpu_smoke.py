"""Strict pretrained component smoke on synthetic inputs, NOT medical evaluation."""
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from baseline_contracts import validate_latents


def main():
    root = Path(__file__).resolve().parent / "assets"
    if not (root / "DOWNLOAD_COMPLETE").exists():
        raise RuntimeError("Assets incomplete")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    device = "cuda"
    dtype = torch.float16
    started = time.perf_counter()
    # from_pretrained uses maintained safetensors loaders, not pickle checkpoints.
    unet, info = UNet2DConditionModel.from_pretrained(
        str(root / "smile/SMILE"), subfolder="unet", local_files_only=True,
        use_safetensors=True, torch_dtype=dtype, output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Non-strict checkpoint load: {info}")
    unet = unet.eval().to(device)
    vae = AutoencoderKL.from_pretrained(str(root / "smile/autoencoder"),
        subfolder="vae", local_files_only=True, use_safetensors=True,
        torch_dtype=dtype).eval().to(device)
    text = CLIPTextModel.from_pretrained(str(root / "sd15/text_encoder"),
        local_files_only=True, use_safetensors=True, torch_dtype=dtype).eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained(str(root / "sd15/tokenizer"), local_files_only=True)
    scheduler = DDIMScheduler.from_pretrained(str(root / "sd15/scheduler"), local_files_only=True)
    scheduler.set_timesteps(2, device=device)
    generator = torch.Generator(device=device).manual_seed(1729)
    with torch.inference_mode():
        ids = tokenizer("venous phase", padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        context = text(ids)[0]
        # Synthetic normalized triplet: proves component wiring only, not CT realism.
        synthetic = torch.zeros((1, 3, 512, 512), device=device, dtype=dtype)
        source = vae.encode(synthetic).latent_dist.sample(generator=generator) * vae.config.scaling_factor
        noisy = torch.randn(source.shape, device=device, dtype=dtype, generator=generator)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        forward_seconds = []
        for timestep in scheduler.timesteps:
            validate_latents(noisy, source)
            tick = time.perf_counter()
            residual = unet(torch.cat((noisy, source), dim=1), timestep,
                            encoder_hidden_states=context).sample
            torch.cuda.synchronize()
            forward_seconds.append(time.perf_counter() - tick)
            validate_latents(noisy, source, residual)
            noisy = scheduler.step(residual, timestep, noisy, eta=0).prev_sample
        decoded = vae.decode(noisy / vae.config.scaling_factor).sample
        if decoded.shape != synthetic.shape or not torch.isfinite(decoded).all():
            raise RuntimeError("Decoded output shape/finiteness failure")
        result = {"scope": "synthetic two-step pretrained component smoke only",
                  "medical_quality_tested": False, "gpu": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "load_encode_seconds": load_seconds,
                  "denoiser_seconds": forward_seconds, "decoded_shape": list(decoded.shape),
                  "decoded_min": decoded.min().item(), "decoded_max": decoded.max().item(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "elapsed_seconds": time.perf_counter() - started}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
