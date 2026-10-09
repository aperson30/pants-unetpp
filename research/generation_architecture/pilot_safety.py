"""Original preparation helpers; not wired into SMILE or a production trainer."""
import copy
import random

import numpy as np
import torch


DATA_FLAGS = ("patient_linkage_verified", "phase_acquisition_verified",
              "registration_verified", "tumor_annotations_verified", "training_eligible")


def validate_tumor_pilot(records, budget, recipe_frozen=False, safety_margins_frozen=False):
    """Strict tumor-preservation pilot gate, not a gate for all possible training.

    Registration here concerns the paired fidelity endpoint. A separately defined
    relative-z training experiment need not claim voxel-perfect registration.
    A passing manifest does not establish clinical/statistical adequacy.
    """
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not np.isfinite(budget) or budget <= 0:
        raise ValueError("Separate finite positive charged GPU-hour authorization required")
    if recipe_frozen is not True or safety_margins_frozen is not True:
        raise ValueError("Recipe and medical safety criteria must be frozen")
    patients, case_splits, positives = {}, {}, {"train": set(), "development": set()}
    if not records:
        raise ValueError("Empty pilot manifest")
    pairs = set()
    for row in records:
        if any(row.get(flag) is not True for flag in DATA_FLAGS):
            raise ValueError("Unverified pair cannot enter tumor pilot")
        patient, split = row.get("patient"), row.get("split")
        if not isinstance(patient, str) or not patient or split not in positives:
            raise ValueError("Require patient and train/development split")
        if patient in patients and patients[patient] != split:
            raise ValueError("Patient leakage across splits")
        patients[patient] = split
        source, target = row.get("source"), row.get("target")
        if not isinstance(source, str) or not isinstance(target, str) or not source or not target or source == target:
            raise ValueError("Require distinct source/target scan IDs")
        pair = (source, target)
        if pair in pairs:
            raise ValueError("Duplicate pair")
        pairs.add(pair)
        for case in pair:
            if case in case_splits and case_splits[case] != split:
                raise ValueError("Scan leakage across splits")
            case_splits[case] = split
        if row.get("protected_evaluation_overlap_excluded") is not True:
            raise ValueError("Protected PanTS overlap must be excluded")
        if row.get("paired_pancreatic_lesion_verified") is True:
            positives[split].add(patient)
    if not all(positives.values()):
        raise ValueError("Independent tumor-positive patients needed in both splits")
    return {"patients": len(patients), "pairs": len(pairs),
            "tumor_positive_patients": {key: len(value) for key, value in positives.items()},
            "statistical_or_clinical_adequacy_established": False}


def capture_rng():
    name, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"name": name, "keys": keys.tolist(),
            "position": position, "has_gauss": has_gauss, "cached_gauss": cached_gauss},
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    cuda = state["torch_cuda"]
    if cuda is not None and (not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count()):
        raise ValueError("GPU RNG topology changed; do not silently resume")
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32),
                         n["position"], n["has_gauss"], n["cached_gauss"]))
    torch.set_rng_state(state["torch_cpu"])
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def checkpoint(model, optimizer, *, completed_updates, sampler_state,
               recipe_hash, data_hash, scheduler=None, scaler=None, ema=None):
    if type(completed_updates) is not int or completed_updates < 0 or not recipe_hash or not data_hash:
        raise ValueError("Require completed-update index and frozen contract hashes")
    # Snapshot must not retain live parameter/optimizer references.
    return copy.deepcopy({"schema": "generation-pilot-resume-v1", "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "rng": capture_rng(),
        "completed_updates": completed_updates, "sampler_state": sampler_state,
        "recipe_hash": recipe_hash, "data_hash": data_hash,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None})


def resume(state, model, optimizer, *, recipe_hash, data_hash,
           scheduler=None, scaler=None, ema=None):
    if state.get("schema") != "generation-pilot-resume-v1" or state.get("recipe_hash") != recipe_hash or state.get("data_hash") != data_hash:
        raise ValueError("Checkpoint recipe/data contract differs")
    components = {"scheduler": scheduler, "scaler": scaler, "ema": ema}
    if any((state[name] is None) != (component is None) for name, component in components.items()):
        raise ValueError("Checkpoint component set differs")
    cuda_rng = state["rng"]["torch_cuda"]
    if cuda_rng is not None and (not torch.cuda.is_available() or len(cuda_rng) != torch.cuda.device_count()):
        raise ValueError("GPU RNG topology changed; reject before loading model")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    for name, component in components.items():
        if component is not None:
            component.load_state_dict(state[name])
    # Restore last: constructing/loading the model must not consume resumed draws.
    restore_rng(state["rng"])
    return state["completed_updates"], copy.deepcopy(state["sampler_state"])
