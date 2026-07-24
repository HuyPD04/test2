from __future__ import annotations

import numpy as np

from anchor_zoom_rl.rl.replay import NStepAccumulator, Transition


def _transition(reward: float, done: bool = False) -> Transition:
    return Transition(
        state=np.asarray([0.0], dtype=np.float32),
        action=0,
        reward=reward,
        next_state=np.asarray([1.0], dtype=np.float32),
        done=done,
        next_mask=np.asarray([True], dtype=bool),
    )


def test_n_step_accumulator_flushes_terminal_suffixes() -> None:
    accumulator = NStepAccumulator(n_step=3, gamma=0.5)
    assert accumulator.append(_transition(1.0)) == []
    assert accumulator.append(_transition(2.0)) == []
    ready = accumulator.append(_transition(4.0, done=True))

    assert len(ready) == 3
    assert ready[0].reward == 3.0
    assert ready[1].reward == 4.0
    assert ready[2].reward == 4.0
    assert all(item.discount == 0.0 for item in ready)


def test_n_step_keeps_auxiliary_target_from_first_state() -> None:
    first = _transition(1.0)
    first.hard_targets = np.asarray([1.0, 0.0], dtype=np.float32)
    first.hard_target_mask = np.asarray([True, False])
    second = _transition(2.0, done=True)
    second.hard_targets = np.asarray([0.0, 1.0], dtype=np.float32)
    second.hard_target_mask = np.asarray([False, True])
    accumulator = NStepAccumulator(n_step=2, gamma=0.5)

    assert accumulator.append(first) == []
    ready = accumulator.append(second)

    assert ready[0].hard_targets.tolist() == [1.0, 0.0]
    assert ready[0].hard_target_mask.tolist() == [True, False]
