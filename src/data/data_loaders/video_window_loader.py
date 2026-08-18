import logging
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from data.data_loaders.deterministic_shuffle_sampler import DeterministicShuffleSampler
from data.data_loaders.resumable_data_loader import ResumableDataLoader

logger = logging.getLogger(__name__)


def _skip_none_collate(samples: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Stack window tensors, dropping samples that failed to load."""
    valid = [s for s in samples if s is not None]
    if not valid:
        return None
    return torch.stack(valid)


class VideoWindowLoader:
    """Batches fixed-length video windows into ``(B, T, C, H, W)`` tensors.

    Works with any dataset whose ``__getitem__`` returns a
    ``(T, C, H, W)`` float tensor or ``None`` on load failure (both
    ``OpenWorldRunningDataset`` and ``AtariPongDataset`` do). Failed samples
    are dropped at collate time, and batches where every sample failed are
    skipped entirely.

    When ``shuffle=True`` the loader uses ``DeterministicShuffleSampler`` and
    wraps the underlying ``DataLoader`` in a ``ResumableDataLoader`` so
    training can resume mid-epoch without replaying data.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        image_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed

        self.sampler: Optional[DeterministicShuffleSampler] = (
            DeterministicShuffleSampler(
                num_samples=len(dataset),  # type: ignore[arg-type]
                batch_size=batch_size,
                seed=seed,
            )
            if shuffle
            else None
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_skip_none_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        self.resumable_loader = ResumableDataLoader(self.dataloader, self.sampler)

    def create_resumable_loader(
        self, start_epoch: int, start_batch: int
    ) -> ResumableDataLoader:
        """Build a fresh resumable wrapper positioned at a checkpointed batch."""
        return ResumableDataLoader(
            self.dataloader,
            self.sampler,
            start_epoch=start_epoch,
            start_batch=start_batch,
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        for batch in self.resumable_loader:
            if batch is None:
                logger.warning("Skipping batch where every sample failed to load")
                continue
            yield batch

    def __len__(self) -> int:
        return len(self.dataloader)

    def get_state(self) -> Dict[str, Any]:
        return self.resumable_loader.get_state()

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "num_videos": len(self.dataset),  # type: ignore[arg-type]
            "num_batches": len(self),
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "seed": self.seed,
        }
