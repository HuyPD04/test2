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


def test_auxiliary_hard_action_loss_updates_head() -> None:
    cfg = TrainConfig(
        hidden_dim=32,
        batch_size=2,
        min_replay=2,
        replay_size=4,
        hard_aux_loss_weight=0.2,
    )
    agent = DQNAgent(state_dim=4, action_dim=3, cfg=cfg, device="cpu")
    replay = ReplayBuffer(capacity=4)
    before = [
        parameter.detach().clone()
        for parameter in agent.online.hardness.parameters()
    ]
    for target in (0.0, 1.0):
        replay.add(
            Transition(
                state=np.asarray([target, 0.1, 0.2, 0.3], dtype=np.float32),
                action=0,
                reward=0.0,
                next_state=np.asarray([0.3, 0.2, 0.1, target], dtype=np.float32),
                done=True,
                next_mask=np.asarray([False, False, True], dtype=bool),
                discount=0.0,
                hard_targets=np.asarray([target, 0.0, 0.0], dtype=np.float32),
                hard_target_mask=np.asarray([True, True, False]),
            )
        )

    loss = agent.optimize(replay)
    after = list(agent.online.hardness.parameters())

    assert loss is not None
    assert agent.last_hard_aux_loss > 0.0
    assert any(
        not np.array_equal(old.numpy(), new.detach().numpy())
        for old, new in zip(before, after, strict=True)
    )
