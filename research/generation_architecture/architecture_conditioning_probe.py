"""Frozen tiny-pilot conditioning sensitivity; NOT enhancement/tumor quality."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from architecture_learning_pilot import RECIPE_HASH
from architecture_pilot_contract import draw_latents, require_data_hash, validate_engineering_manifest
from conditional_plain_denoiser import ConditionalPlainDenoiser
from conditional_nested_denoiser import ConditionalNestedDenoiser


def main(root, checkpoint_root, output):
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    if output.exists():
        raise FileExistsError("Preserve previous probe")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    data = root / "architecture_pilot_inputs_v2"
    manifest = json.loads((data / "pilot_data.json").read_text())
    validate_engineering_manifest(manifest, .5)
    require_data_hash(data / "triplets.npz", manifest["triplets_sha256"])
    assets = root / "assets"
    scheduler = DDPMScheduler.from_pretrained(str(assets / "sd15/scheduler"), local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(str(assets / "sd15/tokenizer"), local_files_only=True)
    text = CLIPTextModel.from_pretrained(str(assets / "sd15/text_encoder"), local_files_only=True,
                                       use_safetensors=True).eval().requires_grad_(False).cuda()
    with torch.no_grad():
        ids = tokenizer(manifest["phase_prompts"], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.cuda()
        context = text(ids)[0].float().detach()
    del text
    vae = AutoencoderKL.from_pretrained(str(assets / "smile/autoencoder"), subfolder="vae",
                                       local_files_only=True, use_safetensors=True).eval().requires_grad_(False).cuda()
    stats = []
    with np.load(data / "triplets.npz", allow_pickle=False) as arrays, torch.no_grad():
        for role in ("source", "target"):
            array = arrays[f"development_{role}"]
            assert array.shape == (4, 3, 512, 512) and array.dtype == np.float32
            means, stds = [], []
            for index in range(4):
                distribution = vae.encode(torch.from_numpy(array[index:index+1]).cuda()).latent_dist
                means.append(distribution.mean * vae.config.scaling_factor)
                stds.append(distribution.std * vae.config.scaling_factor)
            stats.extend((torch.cat(means), torch.cat(stds)))
    del vae
    result = {"scope": "frozen four-triplet engineering conditioning sensitivity",
              "medical_quality_tested": False, "causal_or_statistical_claim": False,
              "zero_source_is_out_of_distribution": True, "results": {}}
    for name, factory in (("plain", ConditionalPlainDenoiser), ("nested", ConditionalNestedDenoiser)):
        state = torch.load(checkpoint_root / f"{name}_checkpoint.pth", map_location="cpu", weights_only=True)
        if (state["recipe_hash"] != RECIPE_HASH or state["data_hash"] != manifest["triplets_sha256"]
                or state["completed_updates"] != 200):
            raise ValueError("Wrong frozen checkpoint")
        model = factory().cuda().eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)
        rows = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in range(4):
                x, t, ctx, noise = draw_latents(stats, index, 50000 + index, context[0:1], scheduler)
                normal = model(x, t, ctx).sample.float()
                zero = x.clone()
                zero[:, 4:] = 0
                predictions = {"normal": normal, "zero_source": model(zero, t, ctx).sample.float(),
                               "other_phase": model(x, t, context[1:2]).sample.float()}
                row = {"index": index, "timestep": int(t.item())}
                for condition, prediction in predictions.items():
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError("Nonfinite probe prediction")
                    row[condition] = {"epsilon_L1": float((prediction-noise).abs().mean()),
                                      "prediction_change_RMS": float((prediction-normal).square().mean().sqrt())}
                rows.append(row)
        result["results"][name] = rows
        del model, state
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(Path(args.root), Path(args.checkpoint_root), Path(args.output))
