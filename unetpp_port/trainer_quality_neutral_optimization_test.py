"""CPU contract tests for the quality-neutral trainer throughput mixin.

Run after copying the trainer files beside nnUNetTrainer.py in an editable nnU-Net v2 checkout.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerQualityNeutralOptimizationMixin import (
    nnUNetTrainerQualityNeutralOptimizationMixin as Optimized,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBF16NoDeepSupervisionSparseValidation import (
    nnUNetTrainerBF16NoDeepSupervisionSparseValidation,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBF16SparseValidation import (
    nnUNetTrainerBF16SparseValidation,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation import (
    nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusPaperSparseValidation import (
    nnUNetTrainerUNetPlusPlusPaperSparseValidation,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusSparseValidation import (
    nnUNetTrainerUNetPlusPlusSparseValidation,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSparseValidationMixin import (
    nnUNetTrainerSparseValidationMixin,
)


def trainer_shell(network, optimizer):
    return SimpleNamespace(
        device=torch.device('cpu'), network=network, optimizer=optimizer,
        loss=torch.nn.MSELoss(), grad_scaler=None,
    )


def check_optimizer_update_equivalence():
    torch.manual_seed(123)
    baseline_network = torch.nn.Linear(4, 3)
    optimized_network = deepcopy(baseline_network)
    baseline_optimizer = torch.optim.SGD(baseline_network.parameters(), lr=0.03, momentum=0.9)
    optimized_optimizer = torch.optim.SGD(optimized_network.parameters(), lr=0.03, momentum=0.9)
    baseline = trainer_shell(baseline_network, baseline_optimizer)
    optimized = trainer_shell(optimized_network, optimized_optimizer)
    optimized._move_target_to_device = Optimized._move_target_to_device.__get__(optimized)
    optimized.reuse_identical_deep_supervision_targets = False

    batches = [
        {'data': torch.randn(2, 4), 'target': torch.randn(2, 3)} for _ in range(5)
    ]
    for batch in batches:
        reference = nnUNetTrainer.train_step(baseline, deepcopy(batch))['loss']
        candidate = Optimized.train_step(optimized, deepcopy(batch))['loss'].cpu().numpy()
        np.testing.assert_array_equal(reference, candidate)

    for reference, candidate in zip(baseline_network.parameters(), optimized_network.parameters()):
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
        torch.testing.assert_close(reference.grad, candidate.grad, rtol=0, atol=0)
    for reference_parameter, candidate_parameter in zip(
            baseline_network.parameters(), optimized_network.parameters()):
        reference_state = baseline_optimizer.state[reference_parameter]
        candidate_state = optimized_optimizer.state[candidate_parameter]
        torch.testing.assert_close(
            reference_state['momentum_buffer'], candidate_state['momentum_buffer'], rtol=0, atol=0
        )


def reference_counts(prediction, target, num_classes, ignore_label=None):
    onehot = torch.zeros((prediction.shape[0], num_classes, *prediction.shape[1:]), dtype=torch.float32)
    onehot.scatter_(1, prediction[:, None], 1)
    target = target.clone()
    if ignore_label is not None:
        mask = (target != ignore_label).float()
        target[target == ignore_label] = 0
    else:
        mask = None
    axes = [0] + list(range(2, onehot.ndim))
    return get_tp_fp_fn_tn(onehot, target, axes=axes, mask=mask)[:3]


def check_confusion_counts():
    torch.manual_seed(456)
    for ignore_label in (None, 255):
        prediction = torch.randint(0, 29, (4, 7, 9, 11))
        target = torch.randint(0, 29, (4, 1, 7, 9, 11))
        target[0, 0, 0, 0, 0] = 28  # explicitly retain the rare PanTS tumour class in the test
        if ignore_label is not None:
            target[1, 0, 1:3, 2:5, 3:7] = ignore_label
        expected = reference_counts(prediction, target, 29, ignore_label)
        actual = Optimized._exclusive_label_confusion_counts(prediction, target, 29, ignore_label)
        for expected_count, actual_count in zip(expected, actual):
            torch.testing.assert_close(expected_count.long(), actual_count, rtol=0, atol=0)

    # Degenerate cases catch empty-class and all-background behavior.
    prediction = torch.zeros((2, 3, 4, 5), dtype=torch.long)
    target = torch.zeros((2, 1, 3, 4, 5), dtype=torch.long)
    expected = reference_counts(prediction, target, 29)
    actual = Optimized._exclusive_label_confusion_counts(prediction, target, 29)
    for expected_count, actual_count in zip(expected, actual):
        torch.testing.assert_close(expected_count.long(), actual_count, rtol=0, atol=0)


def check_target_reuse_guards():
    shell = SimpleNamespace(
        device=torch.device('cpu'), reuse_identical_deep_supervision_targets=True,
        _get_deep_supervision_scales=lambda: [[1, 1, 1], [1, 1, 1]],
    )
    target = torch.randint(0, 29, (2, 1, 3, 4, 5))
    moved = Optimized._move_target_to_device(shell, [target, target.clone()])
    assert moved[0] is moved[1]
    assert shell._identical_ds_targets_verified

    invalid = SimpleNamespace(
        device=torch.device('cpu'), reuse_identical_deep_supervision_targets=True,
        _get_deep_supervision_scales=lambda: [[1, 1, 1], [1, 1, 1]],
    )
    try:
        Optimized._move_target_to_device(invalid, [target, target + 1])
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-identical target reuse was not rejected")


def check_sparse_variants():
    variants = (
        nnUNetTrainerBF16SparseValidation,
        nnUNetTrainerBF16NoDeepSupervisionSparseValidation,
        nnUNetTrainerUNetPlusPlusSparseValidation,
        nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation,
        nnUNetTrainerUNetPlusPlusPaperSparseValidation,
    )
    assert all(issubclass(variant, nnUNetTrainerSparseValidationMixin) for variant in variants)


if __name__ == '__main__':
    check_optimizer_update_equivalence()
    check_confusion_counts()
    check_target_reuse_guards()
    check_sparse_variants()
    print("Quality-neutral trainer contracts passed: updates, counts, target guards, sparse variants.")
