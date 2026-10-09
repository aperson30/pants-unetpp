"""Two-seed near-capacity controls on tiny abdominal candidates; NOT clinical proof."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from architecture_learning_pilot import atomic_save
from architecture_pilot_contract import draw_latents, require_data_hash, validate_engineering_manifest
from conditional_nested_denoiser import ConditionalNestedDenoiser
from conditional_plain_denoiser import ConditionalPlainDenoiser
from controlled_pilot_core import ARMS, SEEDS, UPDATES, intervene, predicted_clean, validate_arrays
from pilot_safety import checkpoint, resume

RECIPE = {"updates": UPDATES, "seeds": SEEDS, "arms": ARMS, "learning_rate": 1e-5,
          "accumulation": 2, "weight_decay": .01, "loss": "epsilon L1", "clip": 1,
          "source": "only four venous abdominal training triplets; provisional separate development group",
          "scope": "engineering conditioning and near-capacity feasibility, no clinical claims",
          "precision": "FP32 parameters/BF16 autocast", "validation_t": [100, 500, 900]}
RECIPE_HASH = hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()


def main(root, output):
    from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from PIL import Image, ImageDraw
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    def guard():
        if time.monotonic()-started > 300:
            raise TimeoutError("Consolidated pilot deadline; no retry")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    assets, data = root / "assets", root / "architecture_pilot_inputs_v2"
    manifest = json.loads((data / "pilot_data.json").read_text())
    validate_engineering_manifest(manifest, .5)
    require_data_hash(data / "triplets.npz", manifest["triplets_sha256"])
    scheduler = DDPMScheduler.from_pretrained(str(assets / "sd15/scheduler"), local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Prediction parameterization changed")
    tokenizer = CLIPTokenizer.from_pretrained(str(assets / "sd15/tokenizer"), local_files_only=True)
    text = CLIPTextModel.from_pretrained(str(assets / "sd15/text_encoder"), local_files_only=True,
                                       use_safetensors=True).eval().requires_grad_(False).cuda()
    with torch.no_grad():
        ids = tokenizer([manifest["phase_prompts"][0]], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.cuda()
        context = text(ids)[0].float().detach()
    del text
    vae = AutoencoderKL.from_pretrained(str(assets / "smile/autoencoder"), subfolder="vae",
                                       local_files_only=True, use_safetensors=True).eval().requires_grad_(False).cuda()
    stats, images = {}, {}
    with np.load(data / "triplets.npz", allow_pickle=False) as arrays, torch.no_grad():
        validate_arrays(arrays)
        for split in ("train", "development"):
            pieces = []
            for role in ("source", "target"):
                a = arrays[f"{split}_{role}"][:4]
                images[(split, role)] = torch.from_numpy(a.copy()).cuda()
                means, stds = [], []
                for i in range(4):
                    guard()
                    d = vae.encode(images[(split, role)][i:i+1]).latent_dist
                    means.append(d.mean*vae.config.scaling_factor)
                    stds.append(d.std*vae.config.scaling_factor)
                pieces.extend((torch.cat(means), torch.cat(stds)))
            stats[split] = tuple(pieces)

    def evaluation(model, phase_only):
        rows = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(4):
                for repeat in range(2):
                    for t_value in (100, 500, 900):
                        guard()
                        x, t, ctx, noise, target = draw_latents(stats["development"], i, 70000+100*i+repeat,
                                                              context, scheduler, timestep_override=t_value,
                                                              return_target=True)
                        # Same posterior/noise for all conditions, recompute noisy target at fixed t.
                        alpha = scheduler.alphas_cumprod[t_value].to(x.device)
                        # Different provisional patient, not an adjacent slice from same patient.
                        wrong = draw_latents(stats["train"], i, 80000+i+repeat, context, scheduler)[0][:, 4:]
                        row = {"triplet": i, "repeat": repeat, "timestep": t_value}
                        for condition in ("correct", "mismatched", "absent"):
                            input_x = intervene(x, "absent" if phase_only else condition, wrong)
                            pred = model(input_x, t, ctx).sample.float()
                            if not torch.isfinite(pred).all():
                                raise FloatingPointError("Nonfinite evaluation")
                            clean = predicted_clean(x[:, :4], pred, alpha)
                            row[condition] = {"epsilon_L1": float((pred-noise).abs().mean()),
                                              "clean_latent_L1": float((clean-target).abs().mean())}
                        rows.append(row)
        return rows

    def update(model, optimizer, index, seed, phase_only):
        optimizer.zero_grad(set_to_none=True)
        for acc in range(2):
            micro = index*2+acc
            x, t, ctx, noise = draw_latents(stats["train"], micro%4, seed*100000+micro, context, scheduler)
            if phase_only:
                x = intervene(x, "absent")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x, t, ctx).sample
                loss = torch.nn.functional.l1_loss(pred.float(), noise.float())/2
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite learning")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()

    result = {"recipe": RECIPE, "recipe_hash": RECIPE_HASH, "job_id": os.getenv("SLURM_JOB_ID"),
              "data_hash": manifest["triplets_sha256"], "torch": torch.__version__,
              "medical_quality_tested": False, "patient_independence_certified": False,
              "registration_certified": False, "arms": {}}
    for seed in SEEDS:
        for name, widths in ARMS.items():
            guard()
            torch.manual_seed(seed)
            factory = ConditionalNestedDenoiser if name.startswith("nested") else ConditionalPlainDenoiser
            model = factory(channels=widths).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
            phase_only = name == "plain_phase_only"
            torch.cuda.synchronize()
            tick = time.monotonic()
            times = []
            for index in range(UPDATES):
                guard()
                update(model, optimizer, index, seed, phase_only)
                if (index+1)%250 == 0:
                    torch.cuda.synchronize()
                    times.append({"updates": index+1, "seconds": time.monotonic()-tick})
                    print(json.dumps({"seed": seed, "arm": name, **times[-1]}), flush=True)
            torch.cuda.synchronize()
            training_seconds = time.monotonic()-tick
            model.eval()
            rows = evaluation(model, phase_only)
            state = checkpoint(model, optimizer, completed_updates=UPDATES,
                               sampler_state={"micro_position": UPDATES*2}, recipe_hash=RECIPE_HASH,
                               data_hash=manifest["triplets_sha256"])
            atomic_save(state, output/f"{seed}_{name}.pth")
            # Verify serialized recovery at the new width/update count, preserving
            # the1000-update checkpoint for generation after the replay check.
            update(model, optimizer, UPDATES, seed, phase_only)
            loaded = torch.load(output/f"{seed}_{name}.pth", map_location="cpu", weights_only=True)
            replay = factory(channels=widths).cuda().train()
            replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=1e-5, weight_decay=.01)
            completed, sampler = resume(loaded, replay, replay_optimizer, recipe_hash=RECIPE_HASH,
                                        data_hash=manifest["triplets_sha256"])
            if completed != UPDATES or sampler != {"micro_position": UPDATES*2}:
                raise RuntimeError("Resume index mismatch")
            update(replay, replay_optimizer, UPDATES, seed, phase_only)
            for a, b in zip(model.parameters(), replay.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                if not torch.isfinite(a).all():
                    raise FloatingPointError("Nonfinite learned parameters")
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            del loaded, replay, replay_optimizer
            # One genuine generation chain per arm/seed, fixed starting noise.
            ddim = DDIMScheduler.from_config(scheduler.config)
            ddim.set_timesteps(20, device="cuda")
            generator = torch.Generator(device="cuda").manual_seed(9911)
            latent = torch.randn(stats["development"][0][0:1].shape, device="cuda", generator=generator)
            source = stats["development"][0][0:1]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.cuda.synchronize()
                inference_tick = time.monotonic()
                for t in ddim.timesteps:
                    guard()
                    x = torch.cat((ddim.scale_model_input(latent, t), source), dim=1)
                    if phase_only:
                        x = intervene(x, "absent")
                    pred = model(x, t.expand(1), context).sample.float()
                    latent = ddim.step(pred, t, latent, eta=0).prev_sample
                torch.cuda.synchronize()
                denoising_seconds = time.monotonic()-inference_tick
            with torch.no_grad():
                decoded = vae.decode(latent.float()/vae.config.scaling_factor).sample.float()
                oracle = vae.decode(stats["development"][2][0:1]/vae.config.scaling_factor).sample.float()
            if not torch.isfinite(decoded).all():
                raise FloatingPointError("Nonfinite generated image")
            target_image = images[("development", "target")][0:1]
            source_image = images[("development", "source")][0:1]
            image_metrics = {"generated_target_L1_unregistered": float((decoded-target_image).abs().mean()),
                             "copy_source_target_L1_unregistered": float((source_image-target_image).abs().mean()),
                             "VAE_target_reconstruction_L1": float((oracle-target_image).abs().mean()),
                             "clinical_or_tumor_metric": False}
            canvas = Image.new("RGB", (256*4, 280), "black")
            draw = ImageDraw.Draw(canvas)
            for column, (title, tensor) in enumerate(zip(("source", "candidate target", "VAE target", "generated: engineering only"),
                                                        (source_image, target_image, oracle, decoded))):
                gray = (tensor[0,1].detach().cpu().numpy()*1000+160)/400
                gray = (np.clip(gray,0,1)*255).astype(np.uint8)
                canvas.paste(Image.fromarray(gray.T[::-1]).resize((256,256)), (256*column,24))
                draw.text((256*column+2,2), title, fill="white")
            canvas.save(output/f"{seed}_{name}.png")
            report = {"parameters": sum(p.numel() for p in model.parameters()),
                      "GPU_next_update_resume_bitwise_pass": True,
                      "training_seconds": training_seconds, "learning_timing": times,
                      "denoising20_seconds": denoising_seconds, "metrics": image_metrics,
                      "conditioning_rows": rows}
            key = f"{seed}_{name}"
            result["arms"][key] = report
            with (output/f"{key}_report.json").open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            print(json.dumps({"completed": key, "metrics": image_metrics}), flush=True)
            del state, model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
    result["elapsed_seconds"] = time.monotonic()-started
    with (output/"result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    main(Path(args.root), Path(args.out_dir))
