"""Quality-neutral throughput optimizations for the PanTS comparison trainers.

This mixin defers scalar loss copies until epoch end, computes exclusive-label validation counts
without dense class-expanded tensors, and can reuse verified-identical full-resolution UNet++ deep-
supervision targets. It does not alter samples, augmentation, model outputs, loss, or updates.
"""
from typing import List

import numpy as np
import torch
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast
from torch import distributed as dist


class nnUNetTrainerQualityNeutralOptimizationMixin:
    reuse_identical_deep_supervision_targets = False

    def _move_target_to_device(self, target):
        if not isinstance(target, list):
            return target.to(self.device, non_blocking=True)
        if not self.reuse_identical_deep_supervision_targets:
            return [i.to(self.device, non_blocking=True) for i in target]

        scales = self._get_deep_supervision_scales()
        all_full_resolution = (
            scales is not None
            and len(scales) == len(target)
            and all(all(float(scale) == 1.0 for scale in output_scale) for output_scale in scales)
        )
        if not all_full_resolution:
            raise RuntimeError("Refusing target reuse because not every deep-supervision scale is 1")
        if not getattr(self, "_identical_ds_targets_verified", False):
            if not target or any(not torch.equal(target[0], candidate) for candidate in target[1:]):
                raise RuntimeError("Refusing target reuse because the first target list is not identical")
            self._identical_ds_targets_verified = True

        target_on_device = target[0].to(self.device, non_blocking=True)
        return [target_on_device] * len(target)

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = self._move_target_to_device(batch['target'])

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # A clone prevents compiled/CUDA-graph replay from overwriting an earlier retained scalar.
        return {'loss': loss.detach().clone()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        # One device synchronization per epoch rather than one per optimizer step.
        losses = torch.stack([output['loss'].reshape(()) for output in train_outputs]).cpu().numpy()
        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, losses)
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(losses)
        self.logger.log('train_losses', loss_here, self.current_epoch)

    @staticmethod
    def _exclusive_label_confusion_counts(prediction: torch.Tensor,
                                          target: torch.Tensor,
                                          num_classes: int,
                                          ignore_label=None):
        """Exact TP/FP/FN counts for a single-label segmentation, without dense one-hot tensors."""
        if target.ndim == prediction.ndim + 1:
            if target.shape[1] != 1:
                raise RuntimeError(f"Expected singleton target channel, got {tuple(target.shape)}")
            target = target[:, 0]
        if target.shape != prediction.shape:
            raise RuntimeError(
                f"Prediction and target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )

        prediction = prediction.reshape(-1).long()
        target = target.reshape(-1).long()
        if ignore_label is not None:
            valid = target != int(ignore_label)
            prediction = prediction[valid]
            target = target[valid]

        predicted_count = torch.bincount(prediction, minlength=num_classes)[:num_classes]
        target_count = torch.bincount(target, minlength=num_classes)[:num_classes]
        true_positive = torch.bincount(
            target[prediction == target], minlength=num_classes
        )[:num_classes]

        # Keep integer counts exact through cross-batch/cross-rank aggregation. This also avoids the
        # fp16 overflow possible in the stock dense path for classes with more than 65,504 voxels.
        false_positive = predicted_count - true_positive
        false_negative = target_count - true_positive
        return true_positive, false_positive, false_negative

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = self._move_target_to_device(batch['target'])
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            loss = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            # Multi-label region tasks are not eligible for bincount arithmetic; preserve stock code.
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            if self.label_manager.has_ignore_label:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
            else:
                mask = None
            tp_hard, fp_hard, fn_hard, _ = get_tp_fp_fn_tn(
                predicted_segmentation_onehot, target, axes=axes, mask=mask
            )
        else:
            prediction = output.argmax(1)
            tp_hard, fp_hard, fn_hard = self._exclusive_label_confusion_counts(
                prediction, target, output.shape[1], self.label_manager.ignore_label
            )
            tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]

        return {
            'loss': loss.detach().cpu().numpy(),
            'tp_hard': tp_hard.detach().cpu().numpy(),
            'fp_hard': fp_hard.detach().cpu().numpy(),
            'fn_hard': fn_hard.detach().cpu().numpy(),
        }
