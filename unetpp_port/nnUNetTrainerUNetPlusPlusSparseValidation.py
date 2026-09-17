"""
Runs the full 50-iteration validation block only every VALIDATE_EVERY epochs (plus always on the
final epoch) instead of every epoch, cutting the 250-train/50-val per-epoch iteration count from 300
to an average of 260 -- an idealized ~1.15x loop speedup with ZERO change to any optimizer update, so
final weights are unaffected (aside from incidental RNG/scheduling). This does not need real PanTS
data to reason about correctness -- see nnUNetTrainerUNetPlusPlusSparseValidation_selftest.py.

WHY THIS ISN'T JUST "SKIP THE VALIDATION BLOCK": nnU-Net's own LocalLogger
(nnunetv2/training/logging/nnunet_logger.py) is explicit and aggressive about one invariant:
"YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP" -- every
tracked key (val_losses, mean_fg_dice, dice_per_class_or_region, ...) is a plain list indexed by
epoch, and progress-plotting (`plot_progress_png`) takes the MINIMUM length across all of them to
decide what to draw. Skip a validation call outright and those three lists silently fall behind
train_losses/lrs/timestamps, which either desyncs the progress plot or, worse, could raise on a
length-mismatch assertion the logger checks on the next log() call for that key. This trainer instead
re-logs the previous epoch's validation numbers on skipped epochs, so every list stays exactly one
entry per epoch, at the cost of that epoch's plotted metrics being a flat repeat rather than fresh
data -- which is the correct, intended tradeoff for coarser monitoring, not a bug.

CONSEQUENCE FOR MODEL SELECTION, AND A BUG THE SELF-TEST CAUGHT: the naive assumption was that
re-logging the same mean_fg_dice value on a skipped epoch could never produce a new "best" EMA. That
assumption is FALSE -- ema_fg_dice is `0.9*previous_ema + 0.1*value`, so as long as the repeated
value is above the current (still-catching-up) EMA, repeating it keeps pulling the EMA upward for
several epochs after the real validation that produced it, with zero new information. The self-test
(run with a monotonically increasing fake dice) caught this directly: checkpoint_best kept moving on
every skipped epoch, not just real validation epochs. Fixed by overriding on_epoch_end() to gate the
best-checkpoint comparison behind whether *this* epoch actually validated, tracked via
_last_epoch_was_real_validation (set in run_training, consumed and not otherwise touched here).
`checkpoint_best.pth` now only updates on a real validation epoch, matching Codex's original
research recommendation: use the fixed final checkpoint for comparisons across configs, not
whichever sparse checkpoint happened to win -- comparing configs by which one got luckiest on its
5-epoch-resolution "best" pick would not be a fair comparison.

Label-28 (tumour) onset tracking is also only accurate at 5-epoch resolution under this trainer.
That's an accepted cost, not something to fix here -- checking every epoch defeats the purpose.
"""
from time import time

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join

from .nnUNetTrainerUNetPlusPlus import nnUNetTrainerUNetPlusPlus


class nnUNetTrainerUNetPlusPlusSparseValidation(nnUNetTrainerUNetPlusPlus):

    VALIDATE_EVERY = 5

    def _validation_epoch_due(self, epoch: int) -> bool:
        return (epoch % self.VALIDATE_EVERY == 0) or (epoch == self.num_epochs - 1)

    def _repeat_previous_validation_metrics(self, epoch: int) -> None:
        """Keeps the logger's one-entry-per-epoch invariant on a skipped validation epoch by
        re-logging the last real validation's numbers instead of omitting them."""
        for key in ('val_losses', 'mean_fg_dice', 'dice_per_class_or_region'):
            self.logger.log(key, self.logger.get_value(key, step=-1), epoch)

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            self._last_epoch_was_real_validation = self._validation_epoch_due(epoch)
            with torch.no_grad():
                self.on_validation_epoch_start()
                if self._last_epoch_was_real_validation:
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)
                else:
                    self._repeat_previous_validation_metrics(epoch)

            self.on_epoch_end()

        self.on_train_end()

    def on_epoch_end(self):
        """Identical to nnU-Net's default on_epoch_end, except the 'best' checkpoint comparison is
        gated behind _last_epoch_was_real_validation. Without this, EMA catch-up dynamics let a
        REPEATED validation value keep pulling ema_fg_dice above the previous best for several
        epochs after the real validation that produced it -- caught by
        nnUNetTrainerUNetPlusPlusSparseValidation_selftest.py, which failed against the earlier,
        unguarded version of this method (inherited unmodified from the base trainer)."""
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.get_value('train_losses', step=-1), decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.get_value('val_losses', step=-1), decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.get_value('dice_per_class_or_region', step=-1)])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._last_epoch_was_real_validation:
            if self._best_ema is None or self.logger.get_value('ema_fg_dice', step=-1) > self._best_ema:
                self._best_ema = self.logger.get_value('ema_fg_dice', step=-1)
                self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
