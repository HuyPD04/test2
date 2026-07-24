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
