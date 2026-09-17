"""
Self-test for nnUNetTrainerUNetPlusPlusSparseValidation's run_training loop -- no real data, GPU, or
full nnUNetTrainer construction needed. Drives the REAL run_training / _validation_epoch_due /
_repeat_previous_validation_metrics methods (bound onto a minimal fake trainer) against nnU-Net's
REAL LocalLogger class, and checks the exact invariant its own source is aggressive about: every
logged list must stay exactly one entry per epoch, no matter how many validation epochs get skipped.
"""
import itertools
import types

from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUNetPlusPlusSparseValidation import (
    nnUNetTrainerUNetPlusPlusSparseValidation as SV,
)


class FakeTrainer:
    """Provides only what run_training()/on_epoch_end() actually touch, so the REAL SV methods --
    not reimplementations of them -- are what gets exercised and checked below."""

    def __init__(self, num_epochs, validate_every, tmp_path):
        self.current_epoch = 0
        self.num_epochs = num_epochs
        self.num_iterations_per_epoch = 3
        self.num_val_iterations_per_epoch = 2
        self.VALIDATE_EVERY = validate_every
        self.logger = MetaLogger(output_folder=str(tmp_path), resume=False)
        self._best_ema = None
        self.save_every = 10 ** 9  # never trip the periodic-checkpoint branch in this test
        self.output_folder = str(tmp_path)
        self.local_rank = 0
        self.dataloader_train = itertools.repeat({})
        self.dataloader_val = itertools.repeat({})
        self.validation_calls = []       # epochs where the real val loop actually ran
        self.best_checkpoint_calls = []  # epochs where save_checkpoint('...best.pth') actually ran
        self._fake_dice = iter([0.1 * i for i in range(1, num_epochs + 1)])  # monotonically rising

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', self.current_epoch, self.current_epoch)

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self, train_outputs):
        self.logger.log('train_losses', -1.0, self.current_epoch)
        self.logger.log('lrs', 1e-2, self.current_epoch)

    def on_validation_epoch_start(self):
        pass

    def train_step(self, batch):
        return {}

    def validation_step(self, batch):
        return {}

    def on_validation_epoch_end(self, val_outputs):
        self.validation_calls.append(self.current_epoch)
        dice = next(self._fake_dice)
        self.logger.log('mean_fg_dice', dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', [dice], self.current_epoch)
        self.logger.log('val_losses', -dice, self.current_epoch)

    def print_to_log_file(self, *args, **kwargs):
        pass

    def save_checkpoint(self, filename):
        if filename.endswith('best.pth'):
            self.best_checkpoint_calls.append(self.current_epoch)


def run_case(num_epochs, validate_every, tmp_path):
    trainer = FakeTrainer(num_epochs, validate_every, tmp_path)
    trainer._validation_epoch_due = types.MethodType(SV._validation_epoch_due, trainer)
    trainer._repeat_previous_validation_metrics = types.MethodType(SV._repeat_previous_validation_metrics, trainer)
    trainer.run_training = types.MethodType(SV.run_training, trainer)
    trainer.on_epoch_end = types.MethodType(SV.on_epoch_end, trainer)
    trainer.run_training()
    return trainer


if __name__ == "__main__":
    import os
    import tempfile

    NUM_EPOCHS, VALIDATE_EVERY = 23, 5
    with tempfile.TemporaryDirectory() as tmp_path:
        os.makedirs(tmp_path, exist_ok=True)
        trainer = run_case(NUM_EPOCHS, VALIDATE_EVERY, tmp_path)

        # every tracked list must have EXACTLY one entry per epoch -- this is the invariant nnU-Net's
        # own LocalLogger source warns about breaking
        lengths = {k: len(v) for k, v in trainer.logger.local_logger.my_fantastic_logging.items()}
        assert len(set(lengths.values())) == 1, f"logger list lengths diverged: {lengths}"
        assert list(lengths.values())[0] == NUM_EPOCHS, lengths

        # the real (expensive) validation loop must have run only on the intended epochs: every
        # VALIDATE_EVERY-th epoch, plus the final epoch, and nowhere else
        expected_validation_epochs = sorted(set(
            [e for e in range(NUM_EPOCHS) if e % VALIDATE_EVERY == 0] + [NUM_EPOCHS - 1]
        ))
        assert trainer.validation_calls == expected_validation_epochs, \
            f"expected {expected_validation_epochs}, got {trainer.validation_calls}"

        # checkpoint_best.pth must only ever be written on a real validation epoch. With a
        # monotonically increasing fake dice, every real validation IS a new best, so this should
        # equal expected_validation_epochs exactly -- not just be a subset of it. The earlier,
        # unguarded version of on_epoch_end failed this exact check (best kept moving on skipped
        # epochs via EMA catch-up), which is why on_epoch_end is overridden in the real trainer.
        assert trainer.best_checkpoint_calls == expected_validation_epochs, \
            f"expected best checkpoint on {expected_validation_epochs}, got {trainer.best_checkpoint_calls}"

        fraction_validated = len(expected_validation_epochs) / NUM_EPOCHS
        idealized_speedup = 300 / (250 + 50 * fraction_validated)
        print(f"Sparse-validation self-test passed over {NUM_EPOCHS} epochs "
              f"(validated {len(expected_validation_epochs)}/{NUM_EPOCHS} epochs, "
              f"logger lists all length {NUM_EPOCHS}, checkpoint_best only on real validation epochs, "
              f"idealized loop speedup {idealized_speedup:.3f}x).")
