"""CPU-only comparison of two states emitted by gh200_kernel_screen.py."""
import argparse
import json

import torch


def tensor_diffs(left, right):
    if left.keys() != right.keys():
        raise RuntimeError("state keys differ")
    max_abs = 0.0
    max_reference = 0.0
    allclose = True
    tensor_count = 0
    element_count = 0
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if not torch.is_tensor(left_value):
            continue
        if left_value.shape != right_value.shape:
            raise RuntimeError(f"shape differs for {key}")
        left_float = left_value.float()
        right_float = right_value.float()
        max_abs = max(max_abs, (left_float - right_float).abs().max().item())
        max_reference = max(max_reference, left_float.abs().max().item())
        allclose = allclose and torch.allclose(
            left_float, right_float, rtol=5e-3, atol=2e-5
        )
        tensor_count += 1
        element_count += left_value.numel()
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "max_abs": max_abs,
        "max_reference": max_reference,
        "max_abs_over_max_reference": max_abs / max_reference if max_reference else 0.0,
        "allclose_rtol_5e-3_atol_2e-5": allclose,
    }


def optimizer_tensors(state):
    return {
        f"{parameter_id}:{name}": value
        for parameter_id, parameter_state in state["state"].items()
        for name, value in parameter_state.items()
        if torch.is_tensor(value)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    # These are explicit paths to states produced moments earlier by our own benchmark process.
    # TorchVersion metadata is not in the weights-only allowlist, so the otherwise safer loader
    # cannot read them. Never use this comparator on an untrusted checkpoint.
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    result = {
        "model": tensor_diffs(left["model"], right["model"]),
        "gradients": tensor_diffs(left["gradients"], right["gradients"]),
        "optimizer": tensor_diffs(
            optimizer_tensors(left["optimizer"]), optimizer_tensors(right["optimizer"])
        ),
        "left_final_loss": left["result"]["final_loss"],
        "right_final_loss": right["result"]["final_loss"],
        "loss_abs_diff": abs(
            left["result"]["final_loss"] - right["result"]["final_loss"]
        ),
    }
    result["ok"] = all(
        result[group]["allclose_rtol_5e-3_atol_2e-5"]
        for group in ("model", "gradients", "optimizer")
    )
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
