"""Window-index creator for the HuggingFace ``p-doom/atari-pong-dataset``.

The dataset (downloaded with ``scripts.data.download_atari_pong``) is laid
out as::

    <data_dir>/train/data_0000.array_record
    <data_dir>/test/data_0000.array_record
    ...

Each ArrayRecord record is a pickled dict with ``raw_video`` (``uint8``
bytes of shape ``(sequence_length, 84, 84, 3)``), ``sequence_length`` and
optionally ``actions`` (see the Jasmine dataloader the dataset was published
for). This creator scans the shards and bakes sliding windows of exactly
``num_frames_in_video`` frames (stride ``frame_spacing``, sliding by one
frame) into an ``AtariPongVideoLog``, mirroring what
``OpenWorldRunningDatasetCreator`` does for the Pokémon frames. The window
index is cached as JSON inside ``data_dir`` so repeat runs skip the scan.
"""

import logging
import os
import pickle
import random
from glob import glob
from typing import Literal

import numpy as np
from tqdm import tqdm

from data.datasets.atari_pong.atari_pong_types import (
    AtariPongVideoLog,
    AtariPongWindow,
)

logger = logging.getLogger(__name__)

# The 84x84 ALE color frames put the score readout in the top rows and white
# walls/paddles near the edges. Ball detection for --require-full-gameplay
# only looks at the interior play area so digits, walls and paddles don't
# count as the ball.
SCOREBOARD_ROWS = 14
_PLAY_AREA = (slice(SCOREBOARD_ROWS, 80), slice(12, 72))
_BALL_BRIGHTNESS_THRESHOLD = 200
_MAX_BALL_PIXELS = 40


def _load_record(payload: bytes) -> dict:
    record = pickle.loads(payload)
    if "raw_video" not in record or "sequence_length" not in record:
        raise ValueError(
            "ArrayRecord record is missing 'raw_video'/'sequence_length'; "
            "is this really the p-doom/atari-pong-dataset format?"
        )
    return record


def decode_record_frames(payload: bytes) -> np.ndarray:
    """Decode one episode record into a ``(seq_len, 84, 84, 3)`` uint8 array."""
    record = _load_record(payload)
    frames = np.frombuffer(record["raw_video"], dtype=np.uint8)
    return frames.reshape(record["sequence_length"], 84, 84, 3)


def _frame_has_ball(frame: np.ndarray) -> bool:
    play_area = frame[_PLAY_AREA]
    bright = (play_area >= _BALL_BRIGHTNESS_THRESHOLD).all(axis=-1)
    num_bright = int(bright.sum())
    return 1 <= num_bright <= _MAX_BALL_PIXELS


class AtariPongDatasetCreator:
    def __init__(
        self,
        data_dir: str,
        num_frames_in_video: int,
        limit: int,
        image_size: int,
        frame_spacing: int = 1,
        require_full_gameplay: bool = False,
    ):
        self.data_dir = data_dir
        self.num_frames_in_video = num_frames_in_video
        self.limit = limit
        self.image_size = image_size
        self.frame_spacing = frame_spacing
        self.require_full_gameplay = require_full_gameplay

    def setup(self) -> tuple[AtariPongVideoLog, AtariPongVideoLog]:
        train_shards, test_shards = self._resolve_split_shards()
        train_log = self._get_or_create_window_log("train", train_shards)
        test_log = self._get_or_create_window_log("test", test_shards)
        return train_log, test_log

    def _resolve_split_shards(self) -> tuple[list[str], list[str]]:
        train_shards = sorted(
            glob(os.path.join(self.data_dir, "train", "*.array_record"))
        )
        test_shards = sorted(
            glob(os.path.join(self.data_dir, "test", "*.array_record"))
        )
        if not train_shards:
            # Flat layout fallback: split the shards 90/10 ourselves.
            flat_shards = sorted(
                glob(os.path.join(self.data_dir, "*.array_record"))
            )
            if not flat_shards:
                raise FileNotFoundError(
                    f"No .array_record shards found under {self.data_dir}. "
                    "Download the dataset with "
                    "`python -m scripts.data.download_atari_pong "
                    f"--local_dir {self.data_dir}`."
                )
            split_at = max(1, int(len(flat_shards) * 0.9))
            train_shards = flat_shards[:split_at]
            test_shards = flat_shards[split_at:] or flat_shards[-1:]
        if not test_shards:
            test_shards = train_shards[-1:]
            train_shards = train_shards[:-1] or test_shards
        return train_shards, test_shards

    def _window_log_cache_key(self, split: Literal["train", "test"]) -> str:
        gameplay_tag = "fullgame" if self.require_full_gameplay else "all"
        return os.path.join(
            self.data_dir,
            f"atari_pong_windows_{split}_{self.num_frames_in_video}f"
            f"_spacing_{self.frame_spacing}_{gameplay_tag}_{self.limit}.json",
        )

    def _get_or_create_window_log(
        self, split: Literal["train", "test"], shard_paths: list[str]
    ) -> AtariPongVideoLog:
        cache_key = self._window_log_cache_key(split)
        if os.path.exists(cache_key):
            logger.info(f"Loading cached {split} window index from {cache_key}")
            with open(cache_key, "r") as f:
                return AtariPongVideoLog.model_validate_json(f.read())

        log = self._build_window_log(split, shard_paths)
        with open(cache_key, "w") as f:
            f.write(log.model_dump_json())
        logger.info(f"Cached {split} window index to {cache_key}")
        return log

    def _build_window_log(
        self, split: Literal["train", "test"], shard_paths: list[str]
    ) -> AtariPongVideoLog:
        # Imported lazily so simply importing this module doesn't require the
        # array_record native extension.
        from array_record.python.array_record_data_source import (
            ArrayRecordDataSource,
        )

        span = (self.num_frames_in_video - 1) * self.frame_spacing + 1
        windows: list[AtariPongWindow] = []
        progress_bar = tqdm(
            total=self.limit,
            desc=f"Indexing atari_pong {split} windows",
        )

        for shard_path in shard_paths:
            if len(windows) >= self.limit:
                break
            source = ArrayRecordDataSource([shard_path])
            for record_index in range(len(source)):
                if len(windows) >= self.limit:
                    break
                record = _load_record(source[record_index])
                sequence_length = int(record["sequence_length"])
                if sequence_length < span:
                    continue

                valid_starts = self._valid_window_starts(record, sequence_length, span)
                for start_frame in valid_starts:
                    windows.append(
                        AtariPongWindow(
                            shard_path=shard_path,
                            record_index=record_index,
                            start_frame=start_frame,
                            num_frames=self.num_frames_in_video,
                            frame_spacing=self.frame_spacing,
                        )
                    )
                    progress_bar.update(1)
                    if len(windows) >= self.limit:
                        break

        progress_bar.close()

        # Windows from one episode overlap heavily; shuffle deterministically
        # so downstream `limit=` slices sample across episodes instead of
        # taking one episode's worth of near-identical windows.
        rng = random.Random(42)
        rng.shuffle(windows)

        logger.info(
            f"Indexed {len(windows)} atari_pong {split} windows from "
            f"{len(shard_paths)} shards (num_frames_in_video="
            f"{self.num_frames_in_video}, frame_spacing={self.frame_spacing}, "
            f"require_full_gameplay={self.require_full_gameplay})"
        )
        return AtariPongVideoLog(windows=windows)

    def _valid_window_starts(
        self, record: dict, sequence_length: int, span: int
    ) -> list[int]:
        num_starts = sequence_length - span + 1
        if not self.require_full_gameplay:
            return list(range(num_starts))

        frames = np.frombuffer(record["raw_video"], dtype=np.uint8).reshape(
            sequence_length, 84, 84, 3
        )
        ball_present = np.array(
            [_frame_has_ball(frame) for frame in frames], dtype=bool
        )

        starts: list[int] = []
        for start in range(num_starts):
            frame_indices = [
                start + i * self.frame_spacing
                for i in range(self.num_frames_in_video)
            ]
            if ball_present[frame_indices].all():
                starts.append(start)
        return starts
