import logging
from typing import Iterator

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class DeterministicShuffleSampler(Sampler[int]):
    """Seeded shuffle sampler with O(1) mid-epoch skip for resuming.

    The permutation for a given ``(seed, epoch)`` pair is always the same, so
    a resumed run that sets the same epoch sees the same ordering as the
    original run. ``set_start_batch`` drops the first
    ``start_batch * batch_size`` indices from iteration, which means
    dataloader workers never load (and discard) data from already-trained
    batches.

    ``__len__`` intentionally reports the full dataset size regardless of the
    current skip so ``len(DataLoader)`` stays constant across resumes.
    """

    def __init__(self, num_samples: int, batch_size: int, seed: int = 0):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_batch(self, start_batch: int) -> None:
        if start_batch < 0:
            raise ValueError(
                f"start_batch must be non-negative, got {start_batch}"
            )
        self.start_batch = start_batch

    def _permutation(self) -> torch.Tensor:
        generator = torch.Generator()
        # Mix the seed and epoch so different epochs get unrelated
        # permutations while staying fully reproducible.
        generator.manual_seed(self.seed * 100003 + self.epoch)
        return torch.randperm(self.num_samples, generator=generator)

    def __iter__(self) -> Iterator[int]:
        permutation = self._permutation()
        skip = min(self.start_batch * self.batch_size, self.num_samples)
        if skip:
            logger.info(
                f"DeterministicShuffleSampler skipping {skip} samples "
                f"(epoch={self.epoch}, start_batch={self.start_batch})"
            )
        return iter(permutation[skip:].tolist())

    def __len__(self) -> int:
        return self.num_samples
