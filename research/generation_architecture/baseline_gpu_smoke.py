"""Strict pretrained component smoke on synthetic inputs, NOT medical evaluation."""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from baseline_contracts import validate_latents


def main(public_demo=False):
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
    scheduler.set_timesteps(200 if public_demo else 2, device=device)
    generator = torch.Generator(device=device).manual_seed(1729)
    with torch.inference_mode():
        prompts = ["", "An venous phase CT slice."] if public_demo else ["venous phase"]
        ids = tokenizer(prompts, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        context = text(ids)[0]
        # Synthetic normalized triplet: proves component wiring only, not CT realism.
        synthetic = torch.zeros((1, 3, 512, 512), device=device, dtype=dtype)
        if public_demo:
            import nibabel as nib
            import numpy as np
            from baseline_contracts import validate_geometry, hu_to_unit
            image = nib.load(root.parent / "demo_inputs/Dataset101/demo_non-contrast/ct.nii.gz")
            validate_geometry(tuple(int(n) for n in image.shape), torch.tensor(image.affine))
            if image.shape != (512, 512, 283):
                raise RuntimeError("Public demo differs from inspected input")
            pixels = np.asanyarray(image.dataobj[:, :, 140:143]).copy()
            synthetic = (hu_to_unit(torch.from_numpy(pixels).float()).permute(2, 0, 1)
                         .unsqueeze(0).to(device=device, dtype=dtype) * 2 - 1)
        source = vae.encode(synthetic).latent_dist.sample(generator=generator) * vae.config.scaling_factor
        noisy = torch.randn(source.shape, device=device, dtype=dtype, generator=generator)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        forward_seconds = []
        for timestep in scheduler.timesteps:
            validate_latents(noisy, source)
            tick = time.perf_counter()
            noisy_input = torch.cat((noisy, noisy), dim=0) if public_demo else noisy
            source_input = torch.cat((source, source), dim=0) if public_demo else source
            noisy_input = scheduler.scale_model_input(noisy_input, timestep)
            residual = unet(torch.cat((noisy_input, source_input), dim=1), timestep,
                            encoder_hidden_states=context).sample
            if public_demo:
                unconditional, conditional = residual.chunk(2)
                residual = unconditional + 7.5 * (conditional - unconditional)
            torch.cuda.synchronize()
            forward_seconds.append(time.perf_counter() - tick)
            validate_latents(noisy, source, residual)
            noisy = scheduler.step(residual, timestep, noisy, eta=0).prev_sample
        decoded = vae.decode(noisy / vae.config.scaling_factor).sample
        if decoded.shape != synthetic.shape or not torch.isfinite(decoded).all():
            raise RuntimeError("Decoded output shape/finiteness failure")
        if public_demo:
            # A triplet array, explicitly NOT a whole-volume NIfTI or scored tumor result.
            output = root.parent / "demo_triplet_venous_unit.npy"
            with output.open("xb") as stream:
                np.save(stream, (decoded.float() / 2 + 0.5).clamp(0, 1).cpu().numpy())
        result = {"scope": "public center-triplet component smoke; NOT released volume pipeline" if public_demo
                            else "synthetic two-step pretrained component smoke only",
                  "medical_quality_tested": False, "gpu": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "load_encode_seconds": load_seconds,
                  "denoiser_seconds": forward_seconds, "decoded_shape": list(decoded.shape),
                  "steps": len(scheduler.timesteps), "guidance_scale": 7.5 if public_demo else 1.0,
                  "decoded_min": decoded.min().item(), "decoded_max": decoded.max().item(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "elapsed_seconds": time.perf_counter() - started}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-demo", action="store_true")
    main(parser.parse_args().public_demo)
