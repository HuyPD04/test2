from __future__ import annotations

import numpy as np

from anchor_zoom_rl.config import TrainConfig
from anchor_zoom_rl.rl.agent import DQNAgent
from anchor_zoom_rl.rl.replay import ReplayBuffer, Transition


def test_double_dqn_update_accepts_action_masks() -> None:
    cfg = TrainConfig(
        hidden_dim=32,
        batch_size=2,
        min_replay=2,
        replay_size=4,
    )
    agent = DQNAgent(state_dim=4, action_dim=3, cfg=cfg, device="cpu")
    replay = ReplayBuffer(capacity=4)
    for action in (0, 1):
        replay.add(
            Transition(
                state=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
                action=action,
                reward=1.0,
                next_state=np.asarray([0.3, 0.2, 0.1, 0.0], dtype=np.float32),
                done=False,
                next_mask=np.asarray([False, True, True], dtype=bool),
                discount=0.95,
            )
        )
    loss = agent.optimize(replay)
    assert loss is not None
    assert np.isfinite(loss)
