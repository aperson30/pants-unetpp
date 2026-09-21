"""Generic sparse-validation schedule shared by every cell of the architecture x DS grid."""
from time import time

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join


class nnUNetTrainerSparseValidationMixin:
    VALIDATE_EVERY = 5

    def _validation_epoch_due(self, epoch: int) -> bool:
        return (epoch % self.VALIDATE_EVERY == 0) or (epoch == self.num_epochs - 1)

    def _repeat_previous_validation_metrics(self, epoch: int) -> None:
        for key in ('val_losses', 'mean_fg_dice', 'dice_per_class_or_region'):
            self.logger.log(key, self.logger.get_value(key, step=-1), epoch)

    def run_training(self):
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            self._last_epoch_was_real_validation = self._validation_epoch_due(epoch)
            with torch.no_grad():
                self.on_validation_epoch_start()
                if self._last_epoch_was_real_validation:
                    val_outputs = []
                    for _ in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)
                else:
                    self._repeat_previous_validation_metrics(epoch)
            self.on_epoch_end()
        self.on_train_end()

    def on_epoch_end(self):
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
