"""Bounded CPU toy-input contracts, NOT real-case quality or GPU parity."""
import argparse
import json
from pathlib import Path

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import MaisiConvolution

from prepare_msd import MODEL_HASH, sha
from run_maisi import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = args.root / 'inputs' / 'autoencoder_v1.pt'
    if sha(path) != MODEL_HASH:
        raise RuntimeError('Checkpoint hash mismatch')
    model = load_model(path)
    torch.manual_seed(43)
    # Split axis stays wide enough even at deepest encoder level.
    x = torch.rand(1, 1, 8, 64, 8)
    with torch.inference_mode():
        mu, sigma = model.encode(x)
        baseline = model.decode(mu)
        official_reconstruct = model.reconstruct(x)
        mean_contract_error = float((baseline-official_reconstruct).abs().max())
        layers = [layer for layer in model.modules() if isinstance(layer, MaisiConvolution)]
        for layer in layers:
            layer.num_splits = 4
        mu4, sigma4 = model.encode(x)
        split_decode_same_latent = model.decode(mu)
        split_full = model.decode(mu4)
        report = dict(protocol='trained weights, CPU FP32 toy input only; NOT medical quality',
                      model_sha256=MODEL_HASH, input_shape=list(x.shape),
                      reconstruct_mean_contract_max_abs=mean_contract_error,
                      split_conv_count=len(layers),
                      splits_1_vs_4_mu_max_abs=float((mu-mu4).abs().max()),
                      splits_1_vs_4_sigma_max_abs=float((sigma-sigma4).abs().max()),
                      splits_1_vs_4_decode_same_latent_max_abs=float((baseline-split_decode_same_latent).abs().max()),
                      splits_1_vs_4_full_reconstruct_max_abs=float((baseline-split_full).abs().max()),
                      normalized_output_global_scale=float(baseline.abs().max()))
        for layer in layers:
            layer.num_splits = 1
        torch.manual_seed(44)
        stage2 = model.encode_stage_2_inputs(x)
        torch.manual_seed(44)
        expected = model.sampling(mu, sigma)
        report['stage2_sample_contract_max_abs'] = float((stage2-expected).abs().max())
        report['sample_vs_mu_latent_max_abs'] = float((stage2-mu).abs().max())
        report['sigma_mean'] = float(sigma.mean())
        if mean_contract_error != 0 or report['stage2_sample_contract_max_abs'] != 0:
            raise RuntimeError('Mean/sample method contracts differ from inspected source')
        if not all(torch.isfinite(t).all() for t in (baseline, split_full, mu, sigma, stage2)):
            raise RuntimeError('Nonfinite CPU result')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
