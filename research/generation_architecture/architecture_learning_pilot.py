"""Small real-CT epsilon-learning pilot, NOT SMILE reproduction or cancer proof."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from architecture_pilot_contract import draw_latents, require_data_hash, validate_engineering_manifest
from conditional_nested_denoiser import ConditionalNestedDenoiser
from conditional_plain_denoiser import ConditionalPlainDenoiser
from pilot_safety import checkpoint, resume

RECIPE = {"scope": "architecture-only exploratory feasibility; no clinical/generalization claim",
          "updates": 200, "microbatch": 1, "accumulation": 2, "loss": "epsilon L1",
          "learning_rate": 1e-5, "weight_decay": .01, "max_grad_norm": 1,
          "precision": "FP32 parameters, BF16 autocast", "channels": [32, 64, 128],
          "initialization": "both original small models from scratch",
          "VAE": "frozen SMILE posterior parameters cached; fresh shared draws",
          "supervision": "deepest only; no auxiliary losses/EMA/early exits/compile"}
RECIPE_HASH = hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()


def atomic_save(state, path):
    if path.exists():
        raise FileExistsError("Preserve checkpoint")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as output:
        torch.save(state, output)
        output.flush()
        os.fsync(output.fileno())
    temporary.rename(path)


def main(data_dir, output, cap):
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    with (data_dir / "pilot_data.json").open() as source:
        manifest = json.load(source)
    validate_engineering_manifest(manifest, cap)
    require_data_hash(data_dir / "triplets.npz", manifest["triplets_sha256"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    root = Path(__file__).resolve().parent / "assets"
    if not (root / "DOWNLOAD_COMPLETE").exists():
        raise RuntimeError("Assets incomplete")
    scheduler = DDPMScheduler.from_pretrained(str(root / "sd15/scheduler"), local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Freeze epsilon parameterization")
    tokenizer = CLIPTokenizer.from_pretrained(str(root / "sd15/tokenizer"), local_files_only=True)
    text = CLIPTextModel.from_pretrained(str(root / "sd15/text_encoder"), local_files_only=True,
                                       use_safetensors=True).eval().requires_grad_(False).cuda()
    with torch.no_grad():
        ids = tokenizer(manifest["phase_prompts"], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.cuda()
        context = text(ids)[0].float().detach()
    del text
    gc.collect()
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(str(root / "smile/autoencoder"), subfolder="vae",
                                       local_files_only=True, use_safetensors=True).eval().requires_grad_(False).cuda()
    stats, phase_indices = {}, {}
    with np.load(data_dir / "triplets.npz", allow_pickle=False) as dataset, torch.no_grad():
        for split, expected_count in (("train", 8), ("development", 4)):
            pieces = []
            for role in ("source", "target"):
                array = dataset[f"{split}_{role}"]
                if array.shape != (expected_count, 3, 512, 512) or array.dtype != np.float32 or not np.isfinite(array).all() or np.abs(array).max() > 1:
                    raise ValueError("Invalid normalized triplet input")
                means, stds = [], []
                for index in range(expected_count):
                    distribution = vae.encode(torch.from_numpy(array[index:index+1]).cuda()).latent_dist
                    means.append((distribution.mean * vae.config.scaling_factor).detach())
                    stds.append((distribution.std * vae.config.scaling_factor).detach())
                pieces.extend((torch.cat(means), torch.cat(stds)))
            stats[split] = tuple(pieces)
            phase = dataset[f"{split}_phase_index"]
            if phase.shape != (expected_count,) or phase.dtype != np.int64 or np.any((phase < 0) | (phase > 1)):
                raise ValueError("Invalid phase-index input")
            phase_indices[split] = phase.copy()
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    preparation_seconds = time.monotonic() - started

    def batch(split, index, seed):
        phase = int(phase_indices[split][index])
        return draw_latents(stats[split], index, seed, context[phase:phase+1], scheduler)

    def evaluate(model):
        values = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in range(4):
                x, t, ctx, noise = batch("development", index, 50000 + index)
                prediction = model(x, t, ctx).sample
                values.append(float(torch.nn.functional.l1_loss(prediction.float(), noise.float())))
        return sum(values) / len(values)

    def update(model, optimizer, update_index):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for accumulation in range(2):
            micro = update_index * 2 + accumulation
            x, t, ctx, noise = batch("train", micro % 8, 172900 + micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(x, t, ctx).sample
                loss = torch.nn.functional.l1_loss(prediction.float(), noise.float()) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite denoising loss")
            loss.backward()
            losses.append(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        return float(sum(losses)), float(norm)

    reports = {}
    for label, factory in (("plain", ConditionalPlainDenoiser), ("nested", ConditionalNestedDenoiser)):
        torch.manual_seed(1729)
        model = factory().cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
        curve = [{"update": 0, "development_denoising_L1": evaluate(model)}]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        for index in range(200):
            if time.monotonic() - started > 450:
                raise TimeoutError("Pilot time cap; no retries")
            loss, norm = update(model, optimizer, index)
            if (index + 1) % 50 == 0:
                curve.append({"update": index + 1, "last_train_L1": loss,
                              "development_denoising_L1": evaluate(model), "gradient_norm": norm})
                print(json.dumps({"architecture": label, **curve[-1]}), flush=True)
        torch.cuda.synchronize()
        learning_seconds = time.monotonic() - tick
        state = checkpoint(model, optimizer, completed_updates=200, sampler_state={"micro_position": 400},
                           recipe_hash=RECIPE_HASH, data_hash=manifest["triplets_sha256"])
        path = output / f"{label}_checkpoint.pth"
        atomic_save(state, path)
        expected_loss, _ = update(model, optimizer, 200)
        # Loading CPU RNG tensors onto CUDA would break RNG restore; map all to CPU.
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        replay = factory().cuda().train()
        replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=1e-5, weight_decay=.01)
        completed, sampler = resume(loaded, replay, replay_optimizer, recipe_hash=RECIPE_HASH,
                                   data_hash=manifest["triplets_sha256"])
        if completed != 200 or sampler != {"micro_position": 400}:
            raise RuntimeError("Resume index differs")
        replay_loss, _ = update(replay, replay_optimizer, 200)
        if expected_loss != replay_loss:
            raise RuntimeError("Resume loss differs")
        for original, resumed in zip(model.parameters(), replay.parameters()):
            torch.testing.assert_close(original, resumed, rtol=0, atol=0)
            if not torch.isfinite(original).all():
                raise FloatingPointError("Nonfinite trained parameter")
        reports[label] = {"parameters": sum(p.numel() for p in model.parameters()),
                          "curve": curve, "learning_seconds_including_periodic_checks": learning_seconds,
                          "GPU_next_update_resume_bitwise_pass": True,
                          "peak_allocated_bytes_including_resume": torch.cuda.max_memory_allocated()}
        del model, optimizer, replay, replay_optimizer, state, loaded
        gc.collect()
        torch.cuda.empty_cache()
    result = {"scope": RECIPE["scope"], "recipe": RECIPE, "recipe_hash": RECIPE_HASH,
              "job_id": os.getenv("SLURM_JOB_ID"), "data_sha256": manifest["triplets_sha256"],
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "preparation_seconds": preparation_seconds, "elapsed_seconds": time.monotonic()-started,
              "results": reports, "medical_quality_tested": False,
              "capacity_matched": False, "paper_speedup_established": False}
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--charged-cap", type=float, required=True)
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.out_dir), args.charged_cap)
