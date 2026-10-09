"""Fail-closed production-preparation gate, not a clinical certification engine."""
import math

from pilot_safety import validate_tumor_pilot


def validate_larger_run(manifest):
    scope = manifest.get("scope")
    if scope not in ("tumor-preserving-production", "architecture-only-development"):
        raise ValueError("Declare the larger-run endpoint explicitly")
    checks = ("architecture_contracts_passed", "GPU_resume_passed", "useful_conditioning_verified",
              "generation_fidelity_verified", "tumor_safety_verified", "source_pins_verified",
              "preemption_checkpoint_verified", "production_trainer_verified")
    if scope == "architecture-only-development":
        checks = tuple(key for key in checks if key != "tumor_safety_verified")
    missing = [key for key in checks if manifest.get(key) is not True]
    if missing:
        raise ValueError("Uncleared larger-run gates: " + ", ".join(missing))
    budget, requested, spent = [manifest.get(key) for key in
                                ("authorized_charged_hours", "requested_max_charged_hours", "already_spent_charged_hours")]
    if any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in (budget, requested, spent)):
        raise ValueError("Require actual finite authorized/requested/spent budget")
    if requested <= 0 or spent+requested > budget:
        raise ValueError("New run exceeds explicit total authorization")
    if manifest.get("quality_margins_predeclared") is not True or manifest.get("method_selection_uses_protected_test") is not False:
        raise ValueError("Freeze development criteria; protected test cannot select the method")
    if scope == "tumor-preserving-production":
        return validate_tumor_pilot(manifest.get("pairs",[]),budget,recipe_frozen=manifest.get("recipe_frozen"),
                                    safety_margins_frozen=manifest.get("quality_margins_predeclared"))
    if any(manifest.get(key) is not True for key in ("recipe_frozen", "general_pair_manifest_verified",
                                                    "patient_split_verified", "protected_overlap_excluded")):
        raise ValueError("Unverified general-pair development training")
    if manifest.get("medical_quality_claimed") is not False:
        raise ValueError("Architecture-only endpoint cannot imply tumor preservation")
    return {"scope":scope,"tumor_preservation_established":False}
