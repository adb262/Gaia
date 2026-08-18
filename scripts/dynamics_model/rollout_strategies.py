"""Autoregressive rollout strategies for the dynamics model.

Used by ``scripts.dynamics_model.inference --mode compare_rollout_strategies``
to compare ways of fighting error accumulation during long rollouts. Every
strategy consumes the same seed frames and action sequence and returns the
full video (seed + generated frames), so outputs are directly comparable
against ground truth and against ``DynamicsModel.rollout`` (the ``standard``
strategy).
"""

import logging
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutStrategy:
    name: str
    label: str
    description: str


ROLLOUT_STRATEGIES: tuple[RolloutStrategy, ...] = (
    RolloutStrategy(
        name="standard",
        label="Standard",
        description="Plain DynamicsModel.rollout, one sample per frame.",
    ),
    RolloutStrategy(
        name="best_of_n",
        label="Best-of-N",
        description=(
            "Sample N candidates per frame and keep the consensus candidate "
            "(smallest total distance to the other candidates)."
        ),
    ),
    RolloutStrategy(
        name="keyframe_denoise",
        label="Keyframe denoise",
        description=(
            "Spend the full denoising budget on periodic keyframes and a "
            "reduced budget in between."
        ),
    ),
    RolloutStrategy(
        name="outlier_resample",
        label="Outlier resample",
        description=(
            "Resample frames whose change from the previous frame is an "
            "outlier versus the rollout's running average delta."
        ),
    ),
)


def _seed_torch_from_rng(rng: random.Random) -> None:
    torch.manual_seed(rng.getrandbits(31))


def _infer_seed_actions(model, seed_video: torch.Tensor) -> torch.Tensor:
    """Infer latent actions for the transitions inside the seed video."""
    if seed_video.shape[1] >= 2:
        encoded = model.action_model.encode(seed_video)
        return model.action_model.get_action_sequence(encoded).long()
    return torch.empty(
        seed_video.shape[0], 0, dtype=torch.long, device=seed_video.device
    )


def _predict_frame(
    model,
    generated: torch.Tensor,
    action: torch.Tensor,
    action_history: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    """One next-frame prediction with the full context/action bookkeeping.

    ``predict_next_frame`` trims the dynamics context to the model window and
    the action history to match, so passing the full generated video plus the
    full action history replicates ``DynamicsModel.rollout`` semantics.
    """
    context_actions = action_history[:, : generated.shape[1] - 1]
    if context_actions.shape[1] == 0:
        context_actions = None
    return model.predict_next_frame(
        generated,
        action,
        max_steps=max_steps,
        context_actions=context_actions,
    )


def _frame_delta_mse(next_frame: torch.Tensor, prev_frame: torch.Tensor) -> float:
    return F.mse_loss(next_frame.float(), prev_frame.float()).item()


def _pick_consensus_candidate(candidates: list[torch.Tensor]) -> torch.Tensor:
    """Return the candidate with the smallest total distance to the others."""
    if len(candidates) == 1:
        return candidates[0]

    stacked = torch.stack(candidates)  # (N, B, C, H, W)
    flattened = stacked.flatten(start_dim=1).float()
    distances = torch.cdist(flattened, flattened)  # (N, N)
    best_index = int(distances.sum(dim=1).argmin().item())
    return candidates[best_index]


@torch.no_grad()
def rollout_with_strategy(
    model,
    seed_video: torch.Tensor,
    actions: torch.Tensor,
    max_steps: int,
    strategy_name: str,
    rng: random.Random,
    *,
    best_of_n: int = 4,
    keyframe_interval: int = 4,
    outlier_delta_multiplier: float = 2.5,
) -> torch.Tensor:
    """Roll out ``actions`` from ``seed_video`` using the named strategy.

    Args:
        model: A ``DynamicsModel`` (exposes ``rollout``/``predict_next_frame``).
        seed_video: ``(B, S, C, H, W)`` ground-truth seed frames.
        actions: ``(B, K)`` latent action tokens, one per generated frame.
        max_steps: Maximum MaskGIT denoising steps per frame.
        strategy_name: One of ``{s.name for s in ROLLOUT_STRATEGIES}``.
        rng: Python RNG that makes strategy-level sampling reproducible.

    Returns:
        ``(B, S + K, C, H, W)`` video containing seed and generated frames.
    """
    known_names = {strategy.name for strategy in ROLLOUT_STRATEGIES}
    if strategy_name not in known_names:
        raise ValueError(
            f"Unknown rollout strategy {strategy_name!r}. "
            f"Available: {sorted(known_names)}"
        )

    action_sequence = model.normalize_rollout_actions(seed_video, actions)

    if strategy_name == "standard":
        _seed_torch_from_rng(rng)
        return model.rollout(seed_video, action_sequence, max_steps)

    seed_actions = _infer_seed_actions(model, seed_video)
    action_history = torch.cat([seed_actions, action_sequence], dim=1)

    generated = seed_video
    num_rollout_steps = action_sequence.shape[1]
    reduced_steps = max(1, max_steps // 2)
    delta_history: list[float] = []

    for step in range(num_rollout_steps):
        action = action_sequence[:, step]
        prev_frame = generated[:, -1]

        if strategy_name == "best_of_n":
            candidates = []
            for _ in range(best_of_n):
                _seed_torch_from_rng(rng)
                candidates.append(
                    _predict_frame(model, generated, action, action_history, max_steps)
                )
            next_frame = _pick_consensus_candidate(candidates)

        elif strategy_name == "keyframe_denoise":
            is_keyframe = step % keyframe_interval == 0
            _seed_torch_from_rng(rng)
            next_frame = _predict_frame(
                model,
                generated,
                action,
                action_history,
                max_steps if is_keyframe else reduced_steps,
            )

        else:  # outlier_resample
            _seed_torch_from_rng(rng)
            next_frame = _predict_frame(
                model, generated, action, action_history, max_steps
            )
            delta = _frame_delta_mse(next_frame, prev_frame)
            if delta_history:
                mean_delta = sum(delta_history) / len(delta_history)
                threshold = outlier_delta_multiplier * mean_delta
                retries = 0
                while delta > threshold and retries < 3:
                    _seed_torch_from_rng(rng)
                    candidate = _predict_frame(
                        model, generated, action, action_history, max_steps
                    )
                    candidate_delta = _frame_delta_mse(candidate, prev_frame)
                    if candidate_delta < delta:
                        next_frame = candidate
                        delta = candidate_delta
                    retries += 1
                if retries:
                    logger.info(
                        f"outlier_resample step {step}: delta {delta:.6f} vs "
                        f"threshold {threshold:.6f} after {retries} retries"
                    )
            delta_history.append(delta)

        generated = torch.cat([generated, next_frame.unsqueeze(1)], dim=1)

    return generated
