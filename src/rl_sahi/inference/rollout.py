from __future__ import annotations

import time

import numpy as np
import torch

from rl_sahi.common.actions import ACTION_NAMES, Action
from rl_sahi.rl.slice_env import SliceEnv


@torch.inference_mode()
def rollout_one_slice(
    policy,
    env: SliceEnv,
    device: torch.device,
    timing: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    state_start = time.perf_counter()
    state = env.reset()
    if timing is not None:
        timing["rollout_state_ms"] = timing.get("rollout_state_ms", 0.0) + (
            time.perf_counter() - state_start
        ) * 1000.0
    expected_dim = int(getattr(policy, "input_dim", state.shape[0]))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"Checkpoint expects state_dim={expected_dim}, but current detection state has {state.shape[0]}. "
            "Regenerate detection caches and retrain the DQN with the current state configuration."
        )
    actions: list[str] = []
    info: dict = {}
    for _ in range(env.env_cfg.max_steps + 1):
        valid_start = time.perf_counter()
        valid_np = env.policy_action_mask()
        if timing is not None:
            timing["rollout_valid_ms"] = timing.get("rollout_valid_ms", 0.0) + (
                time.perf_counter() - valid_start
            ) * 1000.0
        policy_start = time.perf_counter()
        q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
        valid = torch.from_numpy(valid_np).bool().to(device)
        q[:, ~valid] = -torch.inf
        action = Action(int(q.argmax(dim=1).item()))
        if timing is not None:
            timing["rollout_policy_ms"] = timing.get("rollout_policy_ms", 0.0) + (
                time.perf_counter() - policy_start
            ) * 1000.0
        actions.append(ACTION_NAMES[action])
        result = env.step_inference(action, timing=timing)
        state = result.state
        info = result.info
        if result.done:
            break
    return env.roi.copy(), actions, info
