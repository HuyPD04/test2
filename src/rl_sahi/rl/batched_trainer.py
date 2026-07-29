from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from rl_sahi.common.actions import ACTION_NAMES, Action
from rl_sahi.common.boxes import area, as_boxes, intersection_matrix
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.cache import metadata_matches
from rl_sahi.common.data import iter_images
from rl_sahi.common.device import configure_torch_runtime
from rl_sahi.detection.yolo import load_yolo, load_yolo_variants
from rl_sahi.eval.benchmark import BenchmarkConfig, evaluate_rl_sahi_policy, select_benchmark_images
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.rl.checkpoint import build_training_metadata, save_checkpoint
from rl_sahi.rl.dataset import CachedEpisodeDataset
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.replay import PrioritizedReplayBuffer, ReplayBuffer
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_layout import state_layout_from_detection
from rl_sahi.rl.crop_outcome import CropOutcome, CropOutcomeEvaluator

TRAINING_FLOW_VERSION = 5


@dataclass(slots=True)
class TrainConfig:
    episodes: int = 20000
    num_envs: int = 1
    batch_size: int = 64
    replay_size: int = 50000
    gamma: float = 0.95
    lr: float = 1e-4
    min_replay: int = 512
    target_update: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 15000
    guide_prob_start: float = 0.25
    guide_prob_end: float = 0.05
    guide_decay_steps: int = 15000
    n_step: int = 3
    hidden_dim: int = 512
    use_spatial_cnn: bool = True
    double_dqn: bool = True
    dueling: bool = True
    reward_clip: float = 10.0
    optimize_every: int = 2
    preload_cache: bool = True
    seed: int = 42
    allow_sequence_overlap: bool = False
    log_interval: int = 25
    val_split: str = "val"
    eval_interval: int = 500
    eval_episodes: int = 256
    eval_slice_cost_weight: float = 0.05
    eval_benchmark_images: int = 0
    eval_map_weight: float = 1.0
    eval_small_recall_weight: float = 1.5
    eval_fp_cost_weight: float = 0.01
    resume: bool = True
    resume_interval: int = 25
    use_soft_update: bool = True
    tau: float = 0.005
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000
    use_curriculum: bool = True
    curriculum_steps: int = 15000
    use_crop_outcome_reward: bool = True
    crop_use_cache: bool = True
    crop_detection_reward: float = 0.5
    crop_tp_reward: float = 3.0
    crop_fp_penalty: float = 1.5
    crop_empty_penalty: float = 1.2
    crop_no_gain_penalty: float = 1.2
    hard_hit_reward: float = 4.0
    crop_outcome_reward_scale: float = 0.1
    accepted_no_hard_penalty: float = 2.0
    rejected_crop_penalty: float = 0.5
    crop_attempt_penalty: float = 0.25


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    frac = min(float(step) / max(cfg.epsilon_decay_steps, 1), 1.0)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def guide_prob_by_step(step: int, cfg: TrainConfig) -> float:
    frac = min(float(step) / max(cfg.guide_decay_steps, 1), 1.0)
    return cfg.guide_prob_start + frac * (cfg.guide_prob_end - cfg.guide_prob_start)


def select_action(
    policy: QNetwork,
    state: np.ndarray,
    epsilon: float,
    guide_prob: float,
    env: SliceEnv,
    device: torch.device,
) -> Action:
    valid_actions = np.flatnonzero(env.policy_action_mask())
    if len(valid_actions) == 0:
        valid_actions = np.asarray([int(Action.STOP)], dtype=np.int64)
    if random.random() < guide_prob:
        action = env.guided_action()
        if int(action) in set(int(x) for x in valid_actions):
            return action
    if random.random() < epsilon:
        return Action(int(random.choice(valid_actions.tolist())))
    with torch.no_grad():
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = policy(x)
        valid = torch.from_numpy(env.policy_action_mask()).bool().to(device)
        q[:, ~valid] = -torch.inf
        return Action(int(q.argmax(dim=1).item()))


def soft_update(policy: QNetwork, target: QNetwork, tau: float) -> None:
    for target_parameter, policy_parameter in zip(target.parameters(), policy.parameters()):
        target_parameter.data.copy_(
            tau * policy_parameter.data + (1.0 - tau) * target_parameter.data
        )


def _empty_rois() -> np.ndarray:
    return np.zeros((0, 4), dtype=np.float32)


def _stack_rois(rois: list[np.ndarray]) -> np.ndarray:
    return np.stack(rois).astype(np.float32) if rois else _empty_rois()


def _attempt_overlap(roi: np.ndarray, attempted_rois: list[np.ndarray]) -> float:
    if not attempted_rois:
        return 0.0
    previous = _stack_rois(attempted_rois)
    roi_arr = np.asarray(roi, dtype=np.float32).reshape(1, 4)
    intersection = intersection_matrix(roi_arr, previous)[0]
    current_area = max(float(area(roi_arr)[0]), 1.0)
    return float(np.clip(intersection.max() / current_area, 0.0, 1.0))


def _max_slice_attempts(
    env_cfg: EnvConfig,
    infer_cfg: InferenceConfig | None,
    max_slices: int | None = None,
) -> int:
    if infer_cfg is not None and infer_cfg.max_slice_attempts > 0:
        return int(infer_cfg.max_slice_attempts)
    slice_budget = env_cfg.max_slices if max_slices is None else int(max_slices)
    return max(int(slice_budget) * 2, int(slice_budget), 1)


def _terminal_reward_with_crop_outcome(
    base_reward: float,
    outcome: CropOutcome,
    hard_new_hits: int,
    cfg: TrainConfig,
) -> float:
    hard_new_hits = max(int(hard_new_hits), 0)
    crop_scale = max(float(cfg.crop_outcome_reward_scale), 0.0)
    crop_reward = float(outcome.reward) * crop_scale
    negative_crop_reward = min(float(outcome.reward), 0.0) * crop_scale
    attempt_penalty = max(float(getattr(cfg, "crop_attempt_penalty", 0.0)), 0.0)
    hard_reward = float(cfg.hard_hit_reward) * float(hard_new_hits)
    if outcome.accepted:
        return float(base_reward + hard_reward + crop_reward - attempt_penalty)
    return float(
        min(base_reward, 0.0)
        + negative_crop_reward
        - float(cfg.rejected_crop_penalty)
        - attempt_penalty
    )


def optimize(
    policy: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer | PrioritizedReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    double_dqn: bool = True,
    reward_clip: float = 0.0,
) -> float | None:
    if len(replay) < batch_size:
        return None

    def tensor_from_numpy(value, dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype)
        if device.type == "cuda":
            return tensor.pin_memory().to(device, non_blocking=True)
        return tensor.to(device)

    use_per = isinstance(replay, PrioritizedReplayBuffer)
    next_valid_actions = None
    if use_per:
        sample = replay.sample(batch_size)
        if len(sample) == 8:
            states, actions, rewards, next_states, dones, next_valid_actions, indices, weights = sample
        else:
            states, actions, rewards, next_states, dones, indices, weights = sample
        weights_t = tensor_from_numpy(weights, torch.float32)
    else:
        sample = replay.sample(batch_size)
        if len(sample) == 6:
            states, actions, rewards, next_states, dones, next_valid_actions = sample
        else:
            states, actions, rewards, next_states, dones = sample

    states_t = tensor_from_numpy(states, torch.float32)
    actions_t = tensor_from_numpy(actions, torch.long)
    rewards_t = tensor_from_numpy(rewards, torch.float32)
    if reward_clip and reward_clip > 0.0:
        rewards_t = rewards_t.clamp(-float(reward_clip), float(reward_clip))
    next_states_t = tensor_from_numpy(next_states, torch.float32)
    dones_t = tensor_from_numpy(dones, torch.float32)

    q_values = policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_valid_t = None
        if next_valid_actions is not None:
            next_valid_t = tensor_from_numpy(next_valid_actions, torch.bool)
            invalid_rows = ~next_valid_t.any(dim=1)
            if invalid_rows.any():
                next_valid_t[invalid_rows, int(Action.STOP)] = True
        if double_dqn:
            next_policy_q = policy(next_states_t)
            if next_valid_t is not None:
                next_policy_q = next_policy_q.masked_fill(~next_valid_t, -torch.inf)
            next_actions = next_policy_q.argmax(dim=1)
            next_q = target(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_target_q = target(next_states_t)
            if next_valid_t is not None:
                next_target_q = next_target_q.masked_fill(~next_valid_t, -torch.inf)
            next_q = next_target_q.max(dim=1).values
        target_q = rewards_t + gamma * next_q * (1.0 - dones_t)

    td_errors = q_values - target_q
    if use_per:
        element_loss = F.smooth_l1_loss(q_values, target_q, reduction="none")
        loss = (weights_t * element_loss).mean()
        replay.update_priorities(indices, td_errors.detach().cpu().numpy())
    else:
        loss = F.smooth_l1_loss(q_values, target_q)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def _greedy_eval_episode(
    policy: QNetwork,
    det,
    hard,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    device: torch.device,
    target_classes: tuple[int, ...] = (),
    class_mapping: ClassMapping | None = None,
) -> tuple[int, int, int]:
    attempted_rois: list[np.ndarray] = []
    hard_count = len(as_boxes(hard.hard_boxes)) if env_cfg.use_hard_region_reward else 0
    previous_covered = np.zeros((hard_count,), dtype=bool)
    selected_slices = 0
    max_attempts = _max_slice_attempts(env_cfg, None)
    for _attempt_idx in range(max_attempts):
        if selected_slices >= env_cfg.max_slices:
            break
        history = _stack_rois(attempted_rois)
        env = SliceEnv(
            det,
            hard,
            env_cfg=env_cfg,
            state_cfg=state_cfg,
            previous_rois=history,
            overlap_rois=history,
            previous_covered=previous_covered,
            target_classes=target_classes,
            class_mapping=class_mapping,
            seed_rank=_attempt_idx,
        )
        state = env.reset()
        info = {}
        for _ in range(env_cfg.max_steps + 1):
            with torch.no_grad():
                q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
                valid = torch.from_numpy(env.policy_action_mask()).bool().to(device)
                q[:, ~valid] = -torch.inf
                action = Action(int(q.argmax(dim=1).item()))
            result = env.step(action)
            state = result.state
            info = result.info
            if result.done:
                break

        repeat_overlap = _attempt_overlap(env.roi, attempted_rois)
        attempted_rois.append(env.roi.copy())
        previous_covered = env.covered.copy()
        rejected = (
            info.get("stop_due_to_old_overlap", False)
            or info.get("stop_due_to_attempted_overlap", False)
            or info.get("stop_due_to_max_steps", False)
            or info.get("stop_due_to_stalled_roi", False)
        )
        if rejected:
            if repeat_overlap >= 0.95:
                break
            continue
        selected_slices += 1
        if previous_covered.all() and len(previous_covered) > 0:
            break
    return int(previous_covered.sum()), int(len(previous_covered)), selected_slices


def evaluate_policy(
    policy: QNetwork,
    dataset: CachedEpisodeDataset,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    cfg: TrainConfig,
    device: torch.device,
    target_classes: tuple[int, ...] = (),
    class_mapping: ClassMapping | None = None,
) -> dict[str, float]:
    episodes = min(max(int(cfg.eval_episodes), 1), len(dataset))
    covered_total = 0
    hard_total = 0
    slices_total = 0
    for _ in range(episodes):
        det, hard = dataset.random_episode()
        covered, total, slices = _greedy_eval_episode(
            policy,
            det,
            hard,
            env_cfg,
            state_cfg,
            device,
            target_classes=target_classes,
            class_mapping=class_mapping,
        )
        covered_total += covered
        hard_total += total
        slices_total += slices
    recall = float(covered_total / max(hard_total, 1))
    avg_slices = float(slices_total / max(episodes, 1))
    score = recall - float(cfg.eval_slice_cost_weight) * avg_slices / max(
        float(env_cfg.max_slices), 1.0
    )
    return {
        "val_recall": recall,
        "val_slices": avg_slices,
        "val_score": score,
        "val_covered": float(covered_total),
        "val_hard_total": float(hard_total),
    }


def benchmark_score(
    metrics: dict[str, float],
    cfg: TrainConfig,
    env_cfg: EnvConfig,
) -> float:
    crop_cost = (
        float(cfg.eval_slice_cost_weight)
        * metrics["crops_per_image"]
        / max(float(env_cfg.max_slices), 1.0)
    )
    fp_cost = float(cfg.eval_fp_cost_weight) * metrics["fp_per_image"]
    return (
        float(cfg.eval_map_weight) * metrics["mAP50"]
        + float(cfg.eval_small_recall_weight) * metrics["small_recall"]
        - crop_cost
        - fp_cost
    )


def make_crop_outcome_evaluator(
    model,
    image_root: Path,
    label_root: Path | None,
    cache_root: Path,
    split: str,
    cfg: TrainConfig,
    infer_cfg: InferenceConfig | None,
    bench_cfg: BenchmarkConfig | None,
    eval_weights: Path | None,
    eval_use_cache: bool,
) -> CropOutcomeEvaluator | None:
    if not cfg.use_crop_outcome_reward:
        return None
    if model is None or infer_cfg is None:
        raise RuntimeError(
            "Crop outcome reward requires eval_weights and inference config. "
            "Disable train.use_crop_outcome_reward to use hard-region-only training."
        )
    return CropOutcomeEvaluator(
        model=model,
        image_root=image_root,
        label_root=label_root,
        cache_root=cache_root,
        split=split,
        infer_cfg=infer_cfg,
        weights=eval_weights,
        iou_threshold=float(bench_cfg.iou_threshold) if bench_cfg is not None else 0.5,
        use_cache=eval_use_cache,
        detection_reward=cfg.crop_detection_reward,
        tp_reward=cfg.crop_tp_reward,
        fp_penalty=cfg.crop_fp_penalty,
        empty_penalty=cfg.crop_empty_penalty,
        no_gain_penalty=cfg.crop_no_gain_penalty,
    )


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_resume_checkpoint(
    path: Path,
    policy: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    replay: ReplayBuffer | PrioritizedReplayBuffer,
    state_dim: int,
    train_cfg: TrainConfig,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    layout,
    detection_metadata: dict[str, Any] | None,
    training_metadata: dict[str, Any] | None,
    global_step: int,
    episodes_started: int,
    episodes_completed: int,
    best_score: float,
    best_reward: float,
    optimizer_steps: int,
    scheduler_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_type": "rl_sahi_train_resume",
            "version": TRAINING_FLOW_VERSION,
            "policy": policy.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "replay": replay,
            "state_dim": int(state_dim),
            "state_layout": asdict(layout) if layout is not None else None,
            "train_cfg": asdict(train_cfg),
            "env_cfg": asdict(env_cfg),
            "state_cfg": asdict(state_cfg),
            "detection_metadata": detection_metadata,
            "training_metadata": training_metadata or {},
            "actions": {int(k): v for k, v in ACTION_NAMES.items()},
            "global_step": int(global_step),
            "episodes_started": int(episodes_started),
            "episodes_completed": int(episodes_completed),
            "best_score": float(best_score),
            "best_reward": float(best_reward),
            "optimizer_steps": int(optimizer_steps),
            "scheduler_steps": int(scheduler_steps),
            "rng_state": _rng_state(),
        },
        tmp_path,
    )
    tmp_path.replace(path)


@dataclass
class TransitionRecord:
    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_actions: np.ndarray


@dataclass
class PendingSlice:
    roi: np.ndarray
    info: dict
    transitions: list[TransitionRecord]
    hard_new_hits: int
    terminal_valid: bool
    crop_eligible: bool
    repeat_overlap: float
    outcome: CropOutcome | None = None


@dataclass
class EnvWorker:
    episode: int
    det: Any
    hard: Any
    previous_rois: list[np.ndarray]
    attempted_rois: list[np.ndarray]
    previous_covered: np.ndarray
    full_boxes: np.ndarray
    full_scores: np.ndarray
    full_classes: np.ndarray
    slice_boxes_all: list[np.ndarray]
    slice_scores_all: list[np.ndarray]
    slice_classes_all: list[np.ndarray]
    accepted_new_count: int
    current_max_slices: int
    current_max_attempts: int
    slice_idx: int
    attempt_idx: int
    env: SliceEnv
    state: np.ndarray
    current_transitions: list[TransitionRecord]
    pending_slices: list[PendingSlice]
    total_reward: float
    total_steps: int
    accepted_slices: int
    rejected_slices: int
    crop_new_detection_gain_total: int
    crop_new_detection_utility_total: float
    crop_tp_gain_total: int
    crop_fp_gain_total: int
    crop_outcome_reward_total: float
    losses: list[float]
    info: dict
    collection_done: bool
    pending_optimization_count: int
    done: bool


def _push_n_step_slice(
    transitions: list[TransitionRecord],
    replay: ReplayBuffer | PrioritizedReplayBuffer,
    n_step: int,
    gamma: float,
) -> None:
    horizon = max(int(n_step), 1)
    for start in range(len(transitions)):
        reward = 0.0
        final = transitions[start]
        for offset, transition in enumerate(transitions[start : start + horizon]):
            reward += float(transition.reward) * (float(gamma) ** offset)
            final = transition
            if transition.done:
                break
        first = transitions[start]
        replay.push(
            first.state,
            first.action,
            reward,
            final.next_state,
            final.done,
            final.next_valid_actions,
        )


def batched_train_dqn(
    image_root: Path, cache_root: Path, split: str, out_dir: Path, cfg: TrainConfig, env_cfg: EnvConfig, state_cfg: StateConfig,
    limit: int | None = None, device_name: str | None = None, detection_metadata: dict[str, Any] | None = None,
    hard_region_metadata: dict[str, Any] | None = None,
    target_classes: tuple[int, ...] = (), class_mapping: ClassMapping | None = None, label_root: Path | None = None,
    annotation_root: Path | None = None,
    eval_weights: Path | None = None, eval_full_weights: Path | None = None, eval_crop_weights: Path | None = None,
    infer_cfg: InferenceConfig | None = None, bench_cfg: BenchmarkConfig | None = None,
    eval_use_cache: bool = True,
) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    class_mapping = class_mapping or ClassMapping()
    training_metadata = build_training_metadata(target_classes, class_mapping, infer_cfg, bench_cfg)

    dataset = CachedEpisodeDataset(
        image_root=image_root, cache_root=cache_root, split=split, limit=limit,
        preload=cfg.preload_cache, detection_metadata=detection_metadata,
        hard_region_metadata=hard_region_metadata,
    )
    
    val_dataset = None
    if getattr(cfg, "val_split", ""):
        try:
            val_dataset = CachedEpisodeDataset(
                image_root=image_root, cache_root=cache_root, split=cfg.val_split, limit=limit,
                preload=cfg.preload_cache, detection_metadata=detection_metadata,
                hard_region_metadata=hard_region_metadata,
            )
        except FileNotFoundError as exc:
            print(f"[batched_train] validation disabled: {exc}")

    inference_model = None
    benchmark_model = None
    benchmark_full_model = None
    benchmark_crop_model = None
    benchmark_images: list[Path] = []
    if getattr(cfg, "eval_benchmark_images", 0) > 0:
        if eval_weights is None or label_root is None or infer_cfg is None or bench_cfg is None:
            raise RuntimeError("Benchmark validation requires weights, labels, inference config, and benchmark config")
        benchmark_images = select_benchmark_images(
            iter_images(image_root, split=cfg.val_split),
            cfg.eval_benchmark_images,
            sampling=bench_cfg.sampling,
            seed=bench_cfg.seed,
        )
        if not benchmark_images:
            raise FileNotFoundError(f"No images found for benchmark validation split '{cfg.val_split}'")
        inference_model, benchmark_full_model, benchmark_crop_model = load_yolo_variants(
            eval_weights,
            device=infer_cfg.device,
            full_weights=eval_full_weights,
            crop_weights=eval_crop_weights,
        )
        benchmark_model = inference_model
    elif cfg.use_crop_outcome_reward:
        if eval_weights is None or infer_cfg is None:
            raise RuntimeError("Crop outcome reward requires eval_weights and inference config")
        inference_model = load_yolo(eval_weights, device=infer_cfg.device)
    crop_evaluator = make_crop_outcome_evaluator(
        model=inference_model,
        image_root=image_root,
        label_root=label_root,
        cache_root=cache_root,
        split=split,
        cfg=cfg,
        infer_cfg=infer_cfg,
        bench_cfg=bench_cfg,
        eval_weights=eval_weights,
        eval_use_cache=eval_use_cache,
    )

    probe_det = dataset.first_detection()
    probe_env = SliceEnv(probe_det, None, env_cfg=env_cfg, state_cfg=state_cfg, target_classes=target_classes, class_mapping=class_mapping)
    state_dim = int(probe_env.reset().shape[0])
    layout = state_layout_from_detection(probe_det, state_cfg)

    device = configure_torch_runtime(device_name)
    policy = QNetwork(state_dim, hidden_dim=cfg.hidden_dim, layout=layout, use_spatial_cnn=cfg.use_spatial_cnn, dueling=cfg.dueling).to(device)
    target_net = QNetwork(state_dim, hidden_dim=cfg.hidden_dim, layout=layout, use_spatial_cnn=cfg.use_spatial_cnn, dueling=cfg.dueling).to(device)
    target_net.load_state_dict(policy.state_dict())
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.episodes, eta_min=1e-6)

    if cfg.use_per:
        replay: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(capacity=cfg.replay_size, alpha=cfg.per_alpha, beta_start=cfg.per_beta_start, beta_frames=cfg.per_beta_frames)
    else:
        replay = ReplayBuffer(cfg.replay_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    resume_path = out_dir / "resume.pt"

    best_score = -float("inf")
    best_reward = -float("inf")
    global_step = 0
    num_envs = getattr(cfg, "num_envs", 8)
    episodes_started = 0
    episodes_completed = 0
    optimizer_steps = 0
    scheduler_steps = 0
    resume_loaded = False

    if bool(getattr(cfg, "resume", True)) and resume_path.exists():
        resume_data = _torch_load_checkpoint(resume_path)
        if int(resume_data.get("version", -1)) != TRAINING_FLOW_VERSION:
            raise ValueError(
                "Resume checkpoint uses an incompatible state/reward flow. "
                "Start a fresh batched run with --no-resume."
            )
        if not metadata_matches(resume_data.get("detection_metadata"), detection_metadata):
            raise ValueError(
                "Resume checkpoint detection metadata does not match current weights/config. "
                "Start a fresh run with --no-resume after rebuilding caches."
            )
        if not metadata_matches(resume_data.get("training_metadata"), training_metadata):
            raise ValueError(
                "Resume checkpoint class/inference/benchmark metadata does not match current config. "
                "Start a fresh run with --no-resume."
            )
        if int(resume_data.get("state_dim", -1)) != state_dim:
            raise RuntimeError(
                f"Resume checkpoint state_dim={resume_data.get('state_dim')} does not match current state_dim={state_dim}. "
                f"Delete {resume_path} or run with --no-resume."
            )
        component_values = {
            "state.use_spatial_features": (
                resume_data.get("state_cfg", {}).get("use_spatial_features", True),
                state_cfg.use_spatial_features,
            ),
            "state.use_detector_cues": (
                resume_data.get("state_cfg", {}).get("use_detector_cues", True),
                state_cfg.use_detector_cues,
            ),
            "state.use_history": (
                resume_data.get("state_cfg", {}).get("use_history", True),
                state_cfg.use_history,
            ),
            "train.use_crop_outcome_reward": (
                resume_data.get("train_cfg", {}).get("use_crop_outcome_reward", True),
                cfg.use_crop_outcome_reward,
            ),
            "env.use_hard_region_reward": (
                resume_data.get("env_cfg", {}).get("use_hard_region_reward", True),
                env_cfg.use_hard_region_reward,
            ),
            "env.use_cost_overlap_reward": (
                resume_data.get("env_cfg", {}).get("use_cost_overlap_reward", True),
                env_cfg.use_cost_overlap_reward,
            ),
            "env.use_action_mask": (
                resume_data.get("env_cfg", {}).get("use_action_mask", True),
                env_cfg.use_action_mask,
            ),
        }
        mismatches = [
            f"{name}: checkpoint={bool(saved)}, current={bool(current)}"
            for name, (saved, current) in component_values.items()
            if bool(saved) != bool(current)
        ]
        if mismatches:
            raise ValueError(
                "Resume checkpoint uses a different ablation configuration ("
                + "; ".join(mismatches)
                + "). Use a distinct --out-dir and --no-resume."
            )
        resume_actions = resume_data.get("actions")
        if isinstance(resume_actions, dict) and len(resume_actions) != len(ACTION_NAMES):
            raise RuntimeError(
                f"Resume checkpoint has {len(resume_actions)} actions, current code has {len(ACTION_NAMES)}. "
                f"Delete {resume_path} or run with --no-resume."
            )
        policy.load_state_dict(resume_data["policy"])
        target_net.load_state_dict(resume_data["target_net"])
        optimizer.load_state_dict(resume_data["optimizer"])
        _optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(resume_data["scheduler"])
        replay = resume_data["replay"]
        global_step = int(resume_data.get("global_step", 0))
        episodes_completed = int(resume_data.get("episodes_completed", 0))
        episodes_started = episodes_completed
        best_score = float(resume_data.get("best_score", best_score))
        best_reward = float(resume_data.get("best_reward", best_reward))
        optimizer_steps = int(resume_data.get("optimizer_steps", 0))
        scheduler_steps = int(resume_data.get("scheduler_steps", 0))
        _restore_rng_state(resume_data.get("rng_state", {}))
        resume_loaded = True
        print(
            f"[batched_train] resumed {resume_path} "
            f"(completed={episodes_completed}, global_step={global_step}, replay={len(replay)})"
        )

    print(f"[batched_train] num_envs={num_envs}, episodes={cfg.episodes}")
    if episodes_completed >= cfg.episodes:
        print(f"[batched_train] resume checkpoint already completed {episodes_completed}/{cfg.episodes} episodes")

    def reset_worker(episode: int) -> EnvWorker:
        det, hard = dataset.random_episode()
        current_max_slices = env_cfg.max_slices
        if cfg.use_curriculum:
            curriculum_frac = min(float(global_step) / max(cfg.curriculum_steps, 1), 1.0)
            current_max_slices = max(1, int(env_cfg.max_slices * curriculum_frac))
        hard_count = len(as_boxes(hard.hard_boxes)) if env_cfg.use_hard_region_reward else 0
        previous_covered = np.zeros((hard_count,), dtype=bool)
        if crop_evaluator is not None:
            full_boxes, full_scores, full_classes = crop_evaluator.full_predictions(det)
            accepted_new_count = crop_evaluator.initial_new_count(
                full_boxes,
                full_scores,
                full_classes,
                det.image_shape,
            )
        else:
            full_boxes = np.zeros((0, 4), dtype=np.float32)
            full_scores = np.zeros((0,), dtype=np.float32)
            full_classes = np.zeros((0,), dtype=np.float32)
            accepted_new_count = 0
        current_max_attempts = _max_slice_attempts(env_cfg, infer_cfg, current_max_slices)
        env = SliceEnv(
            det,
            hard,
            env_cfg=env_cfg,
            state_cfg=state_cfg,
            previous_rois=np.zeros((0, 4), dtype=np.float32),
            overlap_rois=np.zeros((0, 4), dtype=np.float32),
            previous_covered=previous_covered,
            target_classes=target_classes,
            class_mapping=class_mapping,
            seed_rank=0,
        )
        return EnvWorker(
            episode=episode, det=det, hard=hard, previous_rois=[], attempted_rois=[], previous_covered=previous_covered,
            full_boxes=full_boxes, full_scores=full_scores, full_classes=full_classes,
            slice_boxes_all=[], slice_scores_all=[], slice_classes_all=[],
            accepted_new_count=accepted_new_count, current_max_slices=current_max_slices,
            current_max_attempts=current_max_attempts, slice_idx=0, attempt_idx=0,
            env=env, state=env.reset(), current_transitions=[], pending_slices=[],
            total_reward=0.0, total_steps=0,
            accepted_slices=0, rejected_slices=0, crop_new_detection_gain_total=0,
            crop_new_detection_utility_total=0.0,
            crop_tp_gain_total=0, crop_fp_gain_total=0, crop_outcome_reward_total=0.0,
            losses=[], info={}, collection_done=False, pending_optimization_count=0, done=False
        )

    active_workers = []
    for _ in range(num_envs):
        if episodes_started < cfg.episodes:
            episodes_started += 1
            active_workers.append(reset_worker(episodes_started))

    append_log = resume_loaded and log_path.exists()
    with log_path.open("a" if append_log else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "global_step",
                "episodes_completed",
                "reward",
                "loss",
                "epsilon",
                "steps",
                "slices",
                "current_max_slices",
                "attempts",
                "rejected_slices",
                "covered",
                "hard_total",
                "crop_new_detection_gain",
                "crop_new_detection_utility",
                "crop_tp_gain",
                "crop_fp_gain",
                "crop_outcome_reward",
                "val_recall",
                "val_slices",
                "val_score",
                "val_mAP50",
                "val_small_recall",
                "val_fp_per_image",
                "val_crops",
                "val_benchmark_score",
            ],
        )
        if not append_log:
            writer.writeheader()

        while active_workers:
            collecting_workers = [worker for worker in active_workers if not worker.collection_done]
            if collecting_workers:
                states = [worker.state for worker in collecting_workers]
                valid_masks = [worker.env.policy_action_mask() for worker in collecting_workers]
                epsilon = epsilon_by_step(global_step, cfg)
                guide_prob = guide_prob_by_step(global_step, cfg)
                actions = [Action.STOP] * len(collecting_workers)
                network_indices: list[int] = []

                for index, worker in enumerate(collecting_workers):
                    valid_mask = valid_masks[index]
                    valid_actions = np.flatnonzero(valid_mask)
                    if len(valid_actions) == 0:
                        valid_actions = np.asarray([int(Action.STOP)], dtype=np.int64)
                    if random.random() < guide_prob:
                        guided_action = worker.env.guided_action()
                        if int(guided_action) < len(valid_mask) and bool(valid_mask[int(guided_action)]):
                            actions[index] = guided_action
                            continue
                    if random.random() < epsilon:
                        actions[index] = Action(int(random.choice(valid_actions.tolist())))
                        continue
                    network_indices.append(index)

                if network_indices:
                    batch_states = np.stack([states[index] for index in network_indices])
                    with torch.no_grad():
                        q_values = policy(torch.from_numpy(batch_states).float().to(device))
                        for batch_index, worker_index in enumerate(network_indices):
                            valid = torch.from_numpy(valid_masks[worker_index]).bool().to(device)
                            q_values[batch_index, ~valid] = -torch.inf
                            actions[worker_index] = Action(
                                int(q_values[batch_index].argmax().item())
                            )

                for index, worker in enumerate(collecting_workers):
                    action = actions[index]
                    result = worker.env.step(action)
                    transition = TransitionRecord(
                        state=np.asarray(worker.state, dtype=np.float32).copy(),
                        action=action,
                        reward=float(result.reward),
                        next_state=np.asarray(result.state, dtype=np.float32).copy(),
                        done=bool(result.done),
                        next_valid_actions=worker.env.policy_action_mask().copy(),
                    )
                    worker.current_transitions.append(transition)
                    worker.state = result.state
                    worker.total_steps += 1
                    worker.info = dict(result.info)
                    global_step += 1

                    optimize_every = max(int(cfg.optimize_every), 1)
                    if global_step % optimize_every == 0:
                        worker.pending_optimization_count += 1
                    if (
                        not cfg.use_soft_update
                        and global_step % max(int(cfg.target_update), 1) == 0
                    ):
                        target_net.load_state_dict(policy.state_dict())

                    if not result.done:
                        continue

                    hard_new_hits = int(
                        (worker.env.covered & ~worker.previous_covered).sum()
                    )
                    repeat_overlap = _attempt_overlap(worker.env.roi, worker.attempted_rois)
                    roi = worker.env.roi.copy()

                    # Commit selection history before any crop inference. The next
                    # slice therefore cannot depend on crop accept/reject.
                    worker.attempted_rois.append(roi)
                    worker.previous_covered = worker.env.covered.copy()
                    worker.attempt_idx += 1
                    terminal_valid = not (
                        result.info.get("stop_due_to_old_overlap", False)
                        or result.info.get("stop_due_to_attempted_overlap", False)
                        or result.info.get("stop_due_to_max_steps", False)
                        or result.info.get("stop_due_to_stalled_roi", False)
                    )
                    crop_eligible = bool(
                        crop_evaluator is not None
                        and not crop_evaluator.should_skip_terminal(result.info)
                    )
                    worker.pending_slices.append(
                        PendingSlice(
                            roi=roi,
                            info=dict(result.info),
                            transitions=worker.current_transitions,
                            hard_new_hits=hard_new_hits,
                            terminal_valid=terminal_valid,
                            crop_eligible=crop_eligible,
                            repeat_overlap=repeat_overlap,
                        )
                    )
                    worker.current_transitions = []

                    selected_candidate = crop_eligible if crop_evaluator is not None else terminal_valid
                    if selected_candidate:
                        worker.slice_idx += 1
                    stop_collection = (
                        worker.slice_idx >= worker.current_max_slices
                        or worker.attempt_idx >= worker.current_max_attempts
                        or (not selected_candidate and repeat_overlap >= 0.95)
                        or (
                            crop_evaluator is None
                            and len(worker.previous_covered) > 0
                            and worker.previous_covered.all()
                        )
                    )
                    if stop_collection:
                        worker.collection_done = True
                        continue

                    history = _stack_rois(worker.attempted_rois)
                    worker.env = SliceEnv(
                        worker.det,
                        worker.hard,
                        env_cfg=env_cfg,
                        state_cfg=state_cfg,
                        previous_rois=history,
                        overlap_rois=history,
                        previous_covered=worker.previous_covered,
                        target_classes=target_classes,
                        class_mapping=class_mapping,
                        seed_rank=worker.attempt_idx,
                    )
                    worker.state = worker.env.reset()
                continue

            # Phase 2: all workers have finished selecting ROIs. Run crop YOLO
            # only now, using micro-batches inside CropOutcomeEvaluator.
            pending_crop_items = [
                (worker, pending)
                for worker in active_workers
                for pending in worker.pending_slices
                if pending.crop_eligible
            ]
            if pending_crop_items and crop_evaluator is not None:
                crop_predictions = crop_evaluator.crop_predictions_many(
                    [worker.det.image_path for worker, _pending in pending_crop_items],
                    [pending.roi for _worker, pending in pending_crop_items],
                )
                for (worker, pending), (raw_boxes, raw_scores, raw_classes) in zip(
                    pending_crop_items,
                    crop_predictions,
                ):
                    outcome = crop_evaluator.evaluate_from_predictions(
                        image_path=worker.det.image_path,
                        det=worker.det,
                        full_boxes=worker.full_boxes,
                        full_scores=worker.full_scores,
                        full_classes=worker.full_classes,
                        slice_boxes_parts=worker.slice_boxes_all,
                        slice_scores_parts=worker.slice_scores_all,
                        slice_classes_parts=worker.slice_classes_all,
                        accepted_new_count=worker.accepted_new_count,
                        raw_boxes=raw_boxes,
                        raw_scores=raw_scores,
                        raw_classes=raw_classes,
                    )
                    pending.outcome = outcome
                    pending.info.update(outcome.info())
                    if outcome.accepted:
                        worker.slice_boxes_all.append(outcome.boxes)
                        worker.slice_scores_all.append(outcome.scores)
                        worker.slice_classes_all.append(outcome.classes)
                        worker.accepted_new_count = outcome.accepted_new_count_after

            # Phase 3: apply terminal outcomes, then create replay entries. This
            # preserves the crop reward in every affected n-step return.
            for worker in active_workers:
                for pending in worker.pending_slices:
                    accepted = False
                    if pending.outcome is not None:
                        terminal_transition = pending.transitions[-1]
                        terminal_transition.reward = _terminal_reward_with_crop_outcome(
                            terminal_transition.reward,
                            pending.outcome,
                            pending.hard_new_hits,
                            cfg,
                        )
                        worker.crop_new_detection_gain_total += int(
                            pending.outcome.new_detection_gain
                        )
                        worker.crop_new_detection_utility_total += float(
                            pending.outcome.new_detection_utility
                        )
                        worker.crop_tp_gain_total += int(pending.outcome.tp_gain)
                        worker.crop_fp_gain_total += int(pending.outcome.fp_gain)
                        worker.crop_outcome_reward_total += float(pending.outcome.reward)
                        accepted = bool(pending.outcome.accepted)
                    elif crop_evaluator is None:
                        accepted = bool(
                            pending.terminal_valid
                            and pending.hard_new_hits >= env_cfg.min_new_hits_to_accept
                        )

                    if accepted:
                        worker.previous_rois.append(pending.roi)
                        worker.accepted_slices += 1
                    else:
                        worker.rejected_slices += 1
                    worker.total_reward += sum(
                        float(transition.reward) for transition in pending.transitions
                    )
                    worker.info = pending.info
                    _push_n_step_slice(
                        pending.transitions,
                        replay,
                        cfg.n_step,
                        cfg.gamma,
                    )

            for worker in active_workers:
                for _ in range(worker.pending_optimization_count):
                    if len(replay) < cfg.min_replay:
                        break
                    loss = optimize(
                        policy,
                        target_net,
                        optimizer,
                        replay,
                        cfg.batch_size,
                        cfg.gamma ** max(int(cfg.n_step), 1),
                        device,
                        double_dqn=cfg.double_dqn,
                        reward_clip=cfg.reward_clip,
                    )
                    if loss is None:
                        break
                    optimizer_steps += 1
                    worker.losses.append(loss)
                    if cfg.use_soft_update:
                        soft_update(policy, target_net, cfg.tau)

            for worker in active_workers:
                if worker.losses:
                    scheduler.step()
                    scheduler_steps += 1
                mean_loss = float(np.mean(worker.losses)) if worker.losses else 0.0
                completed_episode = episodes_completed + 1
                row = {
                    "episode": completed_episode,
                    "global_step": global_step,
                    "episodes_completed": completed_episode,
                    "reward": round(worker.total_reward, 6),
                    "loss": round(mean_loss, 6),
                    "epsilon": round(epsilon_by_step(global_step, cfg), 6),
                    "steps": worker.total_steps,
                    "slices": worker.accepted_slices,
                    "current_max_slices": worker.current_max_slices,
                    "attempts": worker.attempt_idx,
                    "rejected_slices": worker.rejected_slices,
                    "covered": int(worker.previous_covered.sum()),
                    "hard_total": int(len(worker.previous_covered)),
                    "crop_new_detection_gain": worker.crop_new_detection_gain_total,
                    "crop_new_detection_utility": round(
                        worker.crop_new_detection_utility_total, 6
                    ),
                    "crop_tp_gain": worker.crop_tp_gain_total,
                    "crop_fp_gain": worker.crop_fp_gain_total,
                    "crop_outcome_reward": round(worker.crop_outcome_reward_total, 6),
                    "val_recall": "",
                    "val_slices": "",
                    "val_score": "",
                    "val_mAP50": "",
                    "val_small_recall": "",
                    "val_fp_per_image": "",
                    "val_crops": "",
                    "val_benchmark_score": "",
                }

                selected_score = None
                if (
                    completed_episode == 1
                    or completed_episode % max(int(cfg.eval_interval), 1) == 0
                ):
                    if val_dataset is not None:
                        metrics = evaluate_policy(
                            policy,
                            val_dataset,
                            env_cfg,
                            state_cfg,
                            cfg,
                            device,
                            target_classes=target_classes,
                            class_mapping=class_mapping,
                        )
                        row["val_recall"] = round(metrics["val_recall"], 6)
                        row["val_slices"] = round(metrics["val_slices"], 6)
                        row["val_score"] = round(metrics["val_score"], 6)
                        selected_score = metrics["val_score"]
                    if (
                        benchmark_model is not None
                        and benchmark_full_model is not None
                        and benchmark_crop_model is not None
                        and infer_cfg is not None
                        and bench_cfg is not None
                        and label_root is not None
                    ):
                        bench_metrics = evaluate_rl_sahi_policy(
                            model=benchmark_model,
                            full_model=benchmark_full_model,
                            crop_model=benchmark_crop_model,
                            policy=policy,
                            device_t=device,
                            weights=eval_weights,
                            images=benchmark_images,
                            image_root=image_root,
                            label_root=label_root,
                            cache_root=cache_root,
                            split=cfg.val_split,
                            infer_cfg=infer_cfg,
                            bench_cfg=bench_cfg,
                            env_cfg=env_cfg,
                            state_cfg=state_cfg,
                            use_cache=eval_use_cache,
                            annotation_root=annotation_root,
                        )
                        selected_score = benchmark_score(bench_metrics, cfg, env_cfg)
                        row["val_mAP50"] = round(bench_metrics["mAP50"], 6)
                        row["val_small_recall"] = round(
                            bench_metrics["small_recall"], 6
                        )
                        row["val_fp_per_image"] = round(
                            bench_metrics["fp_per_image"], 6
                        )
                        row["val_crops"] = round(bench_metrics["crops_per_image"], 6)
                        row["val_benchmark_score"] = round(selected_score, 6)

                if selected_score is not None and selected_score > best_score:
                    best_score = selected_score
                    save_checkpoint(
                        best_path,
                        policy,
                        state_dim,
                        cfg,
                        env_cfg,
                        state_cfg,
                        layout,
                        detection_metadata=detection_metadata,
                        training_metadata=training_metadata,
                    )
                elif val_dataset is None and worker.total_reward > best_reward:
                    best_reward = worker.total_reward
                    save_checkpoint(
                        best_path,
                        policy,
                        state_dim,
                        cfg,
                        env_cfg,
                        state_cfg,
                        layout,
                        detection_metadata=detection_metadata,
                        training_metadata=training_metadata,
                    )

                writer.writerow(row)
                f.flush()
                if completed_episode % cfg.log_interval == 0 or completed_episode == 1:
                    val_msg = ""
                    if row["val_score"] != "":
                        val_msg = (
                            f" val_recall={row['val_recall']} "
                            f"val_slices={row['val_slices']} val_score={row['val_score']}"
                        )
                    if row["val_benchmark_score"] != "":
                        val_msg += (
                            f" val_mAP50={row['val_mAP50']} "
                            f"small_recall={row['val_small_recall']} "
                            f"benchmark_score={row['val_benchmark_score']}"
                        )
                    print(
                        f"[batched_train] ep={completed_episode}/{cfg.episodes} "
                        f"reward={worker.total_reward:.3f} loss={mean_loss:.4f} "
                        f"eps={epsilon_by_step(global_step, cfg):.3f} "
                        f"slices={worker.accepted_slices}/{worker.current_max_slices} "
                        f"rejected={worker.rejected_slices} "
                        f"covered={row['covered']}/{row['hard_total']}{val_msg}"
                    )

                worker.done = True
                episodes_completed = completed_episode
                resume_interval = max(
                    int(getattr(cfg, "resume_interval", cfg.log_interval)), 1
                )
                if bool(getattr(cfg, "resume", True)) and (
                    episodes_completed == 1
                    or episodes_completed % resume_interval == 0
                ):
                    _save_resume_checkpoint(
                        resume_path,
                        policy,
                        target_net,
                        optimizer,
                        scheduler,
                        replay,
                        state_dim,
                        cfg,
                        env_cfg,
                        state_cfg,
                        layout,
                        detection_metadata,
                        training_metadata,
                        global_step,
                        episodes_started,
                        episodes_completed,
                        best_score,
                        best_reward,
                        optimizer_steps,
                        scheduler_steps,
                    )

            next_active_workers = []
            for worker in active_workers:
                if episodes_started < cfg.episodes:
                    episodes_started += 1
                    next_active_workers.append(reset_worker(episodes_started))
            active_workers = next_active_workers
            
    save_checkpoint(
        last_path, policy, state_dim, cfg, env_cfg, state_cfg, layout,
        detection_metadata=detection_metadata,
        training_metadata=training_metadata,
    )
    if bool(getattr(cfg, "resume", True)):
        _save_resume_checkpoint(
            resume_path,
            policy,
            target_net,
            optimizer,
            scheduler,
            replay,
            state_dim,
            cfg,
            env_cfg,
            state_cfg,
            layout,
            detection_metadata,
            training_metadata,
            global_step,
            episodes_started,
            episodes_completed,
            best_score,
            best_reward,
            optimizer_steps,
            scheduler_steps,
        )
    return best_path
