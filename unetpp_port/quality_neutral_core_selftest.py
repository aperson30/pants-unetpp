"""Dependency-light CPU test of the optimization mixin's numerical contracts.

This can run with PyTorch alone. Tiny stand-ins are installed only for the two nnU-Net functions
imported by the mixin; the test implements its own independent reference update and confusion counts.
"""
import contextlib
import importlib
import sys
import types
from copy import deepcopy
from types import SimpleNamespace

import torch


def install_import_stubs():
    dice = types.ModuleType('nnunetv2.training.loss.dice')
    dice.get_tp_fp_fn_tn = lambda *args, **kwargs: None
    helpers = types.ModuleType('nnunetv2.utilities.helpers')
    helpers.dummy_context = contextlib.nullcontext
    for name in (
        'nnunetv2', 'nnunetv2.training', 'nnunetv2.training.loss', 'nnunetv2.utilities'
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['nnunetv2.training.loss.dice'] = dice
    sys.modules['nnunetv2.utilities.helpers'] = helpers


try:
    Optimized = importlib.import_module(
        'nnunetv2.training.nnUNetTrainer.nnUNetTrainerQualityNeutralOptimizationMixin'
    ).nnUNetTrainerQualityNeutralOptimizationMixin
except ModuleNotFoundError:
    install_import_stubs()
    Optimized = importlib.import_module(
        'nnUNetTrainerQualityNeutralOptimizationMixin'
    ).nnUNetTrainerQualityNeutralOptimizationMixin


def reference_step(trainer, batch):
    data = batch['data'].to(trainer.device, non_blocking=True)
    target = batch['target'].to(trainer.device, non_blocking=True)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer.loss(trainer.network(data), target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.network.parameters(), 12)
    trainer.optimizer.step()
    return loss.detach().cpu()


def shell(network, optimizer):
    return SimpleNamespace(
        device=torch.device('cpu'), network=network, optimizer=optimizer,
        loss=torch.nn.MSELoss(), grad_scaler=None,
        reuse_identical_deep_supervision_targets=False,
    )


def check_updates():
    torch.manual_seed(123)
    reference_network = torch.nn.Linear(4, 3)
    optimized_network = deepcopy(reference_network)
    reference_optimizer = torch.optim.SGD(reference_network.parameters(), lr=0.03, momentum=0.9)
    optimized_optimizer = torch.optim.SGD(optimized_network.parameters(), lr=0.03, momentum=0.9)
    reference = shell(reference_network, reference_optimizer)
    optimized = shell(optimized_network, optimized_optimizer)
    optimized._move_target_to_device = Optimized._move_target_to_device.__get__(optimized)

    for _ in range(5):
        batch = {'data': torch.randn(2, 4), 'target': torch.randn(2, 3)}
        expected_loss = reference_step(reference, deepcopy(batch))
        actual_loss = Optimized.train_step(optimized, deepcopy(batch))['loss']
        torch.testing.assert_close(expected_loss, actual_loss, rtol=0, atol=0)

    for expected, actual in zip(reference_network.parameters(), optimized_network.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(expected.grad, actual.grad, rtol=0, atol=0)
        torch.testing.assert_close(
            reference_optimizer.state[expected]['momentum_buffer'],
            optimized_optimizer.state[actual]['momentum_buffer'], rtol=0, atol=0,
        )


def independent_counts(prediction, target, classes, ignore_label=None):
    target = target[:, 0]
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_label is not None:
        valid = target != ignore_label
    tp = torch.zeros(classes, dtype=torch.int64)
    fp = torch.zeros(classes, dtype=torch.int64)
    fn = torch.zeros(classes, dtype=torch.int64)
    for class_id in range(classes):
        predicted_here = (prediction == class_id) & valid
        target_here = (target == class_id) & valid
        tp[class_id] = (predicted_here & target_here).sum()
        fp[class_id] = (predicted_here & ~target_here).sum()
        fn[class_id] = (~predicted_here & target_here).sum()
    return tp, fp, fn


def check_counts():
    torch.manual_seed(456)
    for ignore_label in (None, 255):
        prediction = torch.randint(0, 29, (4, 7, 9, 11))
        target = torch.randint(0, 29, (4, 1, 7, 9, 11))
        target[0, 0, 0, 0, 0] = 28
        if ignore_label is not None:
            target[1, 0, 1:3, 2:5, 3:7] = ignore_label
        expected = independent_counts(prediction, target, 29, ignore_label)
        actual = Optimized._exclusive_label_confusion_counts(prediction, target, 29, ignore_label)
        for expected_count, actual_count in zip(expected, actual):
            torch.testing.assert_close(expected_count, actual_count, rtol=0, atol=0)


def check_target_guards():
    target = torch.randint(0, 29, (2, 1, 3, 4, 5))
    shell_ok = SimpleNamespace(
        device=torch.device('cpu'), reuse_identical_deep_supervision_targets=True,
        _get_deep_supervision_scales=lambda: [[1, 1, 1], [1, 1, 1]],
    )
    moved = Optimized._move_target_to_device(shell_ok, [target, target.clone()])
    assert moved[0] is moved[1]

    shell_bad = SimpleNamespace(
        device=torch.device('cpu'), reuse_identical_deep_supervision_targets=True,
        _get_deep_supervision_scales=lambda: [[1, 1, 1], [1, 1, 1]],
    )
    try:
        Optimized._move_target_to_device(shell_bad, [target, target + 1])
    except RuntimeError:
        pass
    else:
        raise AssertionError('non-identical target reuse was not rejected')


if __name__ == '__main__':
    check_updates()
    check_counts()
    check_target_guards()
    print('Core optimization contracts passed: exact updates, exact counts, guarded target reuse.')
