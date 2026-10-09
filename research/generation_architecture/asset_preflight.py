"""CPU metadata/import check only; never loads tensor payloads or allocates a GPU."""
import argparse
import importlib
import json
from pathlib import Path

from baseline_contracts import inspect_safetensors


def inspect_assets(root):
    root = Path(root)
    if not (root / "DOWNLOAD_COMPLETE").is_file():
        raise ValueError("Download completion marker missing")
    config = json.loads((root / "smile/SMILE/unet/config.json").read_text())
    if (config["in_channels"], config["out_channels"], config["cross_attention_dim"]) != (8, 4, 768):
        raise ValueError("Unexpected SMILE latent/conditioning contract")
    checkpoint = inspect_safetensors(root / "smile/SMILE/unet/diffusion_pytorch_model.safetensors")
    scheduler = json.loads((root / "sd15/scheduler/scheduler_config.json").read_text())
    if scheduler.get("prediction_type", "epsilon") != "epsilon":
        raise ValueError("This experiment currently requires epsilon prediction")
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub"):
        module = importlib.import_module(name)
        versions[name] = {"version": module.__version__, "path": module.__file__}
    # Exercise lazy imports too; importing the package root alone is insufficient.
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    return {"scope": "CPU headers/config/imports only; NOT model loading or enhancement",
            "checkpoint": checkpoint, "versions": versions,
            "latent_contract": [8, 4, 768], "scheduler_prediction": "epsilon"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_assets(args.assets), indent=2))
