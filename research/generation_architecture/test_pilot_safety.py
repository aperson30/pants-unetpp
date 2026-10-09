import copy
import io
import random
import unittest
import numpy as np
import torch
from pilot_safety import DATA_FLAGS, checkpoint, resume, validate_tumor_pilot


class PilotSafetyTests(unittest.TestCase):
    def records(self):
        return [dict.fromkeys(DATA_FLAGS, True) | {"patient": f"p{i}", "split": split,
            "source": f"nc{i}", "target": f"ce{i}", "paired_pancreatic_lesion_verified": True,
            "protected_evaluation_overlap_excluded": True}
            for i, split in enumerate(("train", "development"))]

    def gate(self, records):
        return validate_tumor_pilot(records, .1, True, True)

    def test_minimum_contract_is_not_clinical_adequacy(self):
        self.assertFalse(self.gate(self.records())["statistical_or_clinical_adequacy_established"])

    def test_unverified_and_leakage_rejected(self):
        for mutate in (lambda r: r[0].update(registration_verified=False),
                       lambda r: r[1].update(patient="p0"),
                       lambda r: r[1].update(source="nc0"),
                       lambda r: r[1].update(paired_pancreatic_lesion_verified=False),
                       lambda r: r[0].update(protected_evaluation_overlap_excluded=False)):
            records = self.records()
            mutate(records)
            with self.assertRaises(ValueError):
                self.gate(records)

    def test_budget_and_recipe_required(self):
        for budget in (None, 0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                validate_tumor_pilot(self.records(), budget, True, True)
        with self.assertRaises(ValueError):
            validate_tumor_pilot(self.records(), .1)

    def test_stochastic_update_resume_exact_after_serialization(self):
        torch.manual_seed(91)
        random.seed(91)
        np.random.seed(91)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9)

        def update(m, opt, schedule):
            x = torch.randn(4, 3) * random.random() + float(np.random.random())
            y = torch.randn(4, 2)
            opt.zero_grad(set_to_none=True)
            loss = (m(x) - y).square().mean()
            loss.backward()
            opt.step()
            schedule.step()
            return loss.detach()

        update(model, optimizer, scheduler)
        saved = checkpoint(model, optimizer, completed_updates=1, sampler_state={"position": 7},
                           recipe_hash="recipe", data_hash="data", scheduler=scheduler)
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        expected = update(model, optimizer, scheduler)
        other = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(other.parameters(), lr=.001)
        schedule = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.9)
        position, sampler = resume(saved, other, opt, recipe_hash="recipe", data_hash="data", scheduler=schedule)
        self.assertEqual((position, sampler), (1, {"position": 7}))
        torch.testing.assert_close(expected, update(other, opt, schedule), rtol=0, atol=0)
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        before = copy.deepcopy(other.state_dict())
        with self.assertRaises(ValueError):
            resume(saved, other, opt, recipe_hash="changed", data_hash="data", scheduler=schedule)
        for name in before:
            torch.testing.assert_close(before[name], other.state_dict()[name], rtol=0, atol=0)
