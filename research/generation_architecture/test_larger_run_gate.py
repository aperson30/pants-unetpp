import unittest
from larger_run_gate import validate_larger_run


class LargerGateTests(unittest.TestCase):
    def test_current_engineering_data_cannot_auto_release_production(self):
        with self.assertRaisesRegex(ValueError,"useful_conditioning_verified"):
            validate_larger_run({"scope":"tumor-preserving-production","architecture_contracts_passed":True,"GPU_resume_passed":True})

    def test_budget_cannot_be_implied_by_go_ahead(self):
        manifest = {key:True for key in ("architecture_contracts_passed","GPU_resume_passed",
                    "useful_conditioning_verified","generation_fidelity_verified","tumor_safety_verified",
                    "source_pins_verified","preemption_checkpoint_verified","production_trainer_verified")}
        manifest.update(scope="tumor-preserving-production",authorized_charged_hours=.5, requested_max_charged_hours=1,
                        already_spent_charged_hours=.03)
        with self.assertRaisesRegex(ValueError,"authorization"):
            validate_larger_run(manifest)

    def test_general_architecture_training_does_not_require_tumor_pairs(self):
        manifest = {key:True for key in ("architecture_contracts_passed","GPU_resume_passed",
                    "useful_conditioning_verified","generation_fidelity_verified","source_pins_verified",
                    "preemption_checkpoint_verified","production_trainer_verified","quality_margins_predeclared",
                    "recipe_frozen","general_pair_manifest_verified","patient_split_verified","protected_overlap_excluded")}
        manifest.update(scope="architecture-only-development",authorized_charged_hours=.5,
                        requested_max_charged_hours=.1,already_spent_charged_hours=.2,
                        method_selection_uses_protected_test=False,medical_quality_claimed=False)
        self.assertFalse(validate_larger_run(manifest)["tumor_preservation_established"])
        manifest["medical_quality_claimed"] = True
        with self.assertRaises(ValueError):
            validate_larger_run(manifest)
