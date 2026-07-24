from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..config import MethodConfig
from ..core.anchors import generate_anchors
from ..core.postprocess import crop_utility, merge_detections
from ..core.reward import crop_step_outcome, match_stats
from ..core.state import state_dimension
from ..core.types import Detections
from ..rl.agent import DQNAgent
from ..rl.replay import NStepAccumulator, ReplayBuffer, Transition
from .data import iter_images, label_path_for, load_yolo_labels, read_image
from .environment import AnchorZoomEnvironment
from .prediction import DetectionRunner


@dataclass(slots=True)
class EpisodeSummary:
    reward: float
    crops: int
    accepted: int
    tp_gain: int
    hard_tp_gain: int
    small_tp_gain: int
    stopped: bool
    loss: float | None
    full_ms: float
    crop_ms: float


class AnchorZoomTrainer:
    def __init__(
        self,
        cfg: MethodConfig,
        train_split: str = "train",
        val_split: str = "val",
        limit: int | None = None,
        runner: DetectionRunner | None = None,
    ) -> None:
        self.cfg = cfg
        self.train_split = train_split
        self.val_split = val_split
        self.images = iter_images(cfg.paths.image_root, train_split, limit)
        if not self.images:
            raise ValueError(f"No training images found in split '{train_split}'")
        self.val_images = iter_images(cfg.paths.image_root, val_split, cfg.train.eval_images)
        self.runner = runner or DetectionRunner(cfg)
        self.agent = DQNAgent(
            state_dimension(cfg.anchors),
            cfg.action_count,
            cfg.train,
            cfg.detector.device,
        )
        self.replay = ReplayBuffer(cfg.train.replay_size)
        self.environment_steps = 0
        self.start_episode = 1
        self.best_score = float("-inf")
        self.rng = random.Random(cfg.train.seed)
        self.output_dir = cfg.paths.output_dir
        self.latest_checkpoint = self.output_dir / "checkpoints" / "latest.pt"
        self.log_path = self.output_dir / "train.csv"
        _set_seed(cfg.train.seed)
        self._prepare_output()
        if cfg.train.resume:
            self._resume()

    def train(self, episodes: int | None = None) -> Path:
        total_episodes = int(episodes or self.cfg.train.episodes)
        window: list[EpisodeSummary] = []
        for episode in range(self.start_episode, total_episodes + 1):
            image_path = self.rng.choice(self.images)
            epsilon = self.agent.epsilon(self.environment_steps)
            summary = self._run_episode(
                image_path=image_path,
                split=self.train_split,
                epsilon=epsilon,
                learn=True,
            )
            window.append(summary)
            self._append_log(episode, image_path, epsilon, summary)

            if episode % max(self.cfg.train.log_interval, 1) == 0:
                recent = window[-self.cfg.train.log_interval :]
                print(
                    f"[train] episode={episode}/{total_episodes} "
                    f"reward={np.mean([item.reward for item in recent]):.3f} "
                    f"crops={np.mean([item.crops for item in recent]):.2f} "
                    f"accepted={np.mean([item.accepted for item in recent]):.2f} "
                    f"epsilon={epsilon:.3f} replay={len(self.replay)} "
                    f"loss={_mean_loss(recent):.4f}"
                )

            if episode % max(self.cfg.train.eval_interval, 1) == 0 and self.val_images:
                score = self.evaluate()
                if score > self.best_score:
                    self.best_score = score
                    self.agent.save(
                        self.cfg.paths.checkpoint,
                        episode,
                        self.environment_steps,
                        best_score=self.best_score,
                    )
                    print(f"[train] new best validation score={score:.4f}")

            if episode % max(self.cfg.train.checkpoint_interval, 1) == 0:
                self.agent.save(
                    self.latest_checkpoint,
                    episode,
                    self.environment_steps,
                    replay=self.replay,
                    best_score=self.best_score,
                )

        final_episode = max(total_episodes, self.start_episode - 1)
        self.agent.save(
            self.latest_checkpoint,
            final_episode,
            self.environment_steps,
            replay=self.replay,
            best_score=self.best_score,
        )
        if not self.cfg.paths.checkpoint.exists():
            self.agent.save(
                self.cfg.paths.checkpoint,
                final_episode,
                self.environment_steps,
                best_score=self.best_score,
            )
        elif not self.cfg.train.resume and self.best_score == float("-inf"):
            self.agent.save(
                self.cfg.paths.checkpoint,
                final_episode,
                self.environment_steps,
                best_score=self.best_score,
            )
        return self.cfg.paths.checkpoint

    def evaluate(self) -> float:
        summaries = [
            self._run_episode(path, self.val_split, epsilon=0.0, learn=False)
            for path in self.val_images
        ]
        score = float(
            np.mean(
                [
                    item.tp_gain
                    + item.hard_tp_gain
                    + 0.5 * item.small_tp_gain
                    - self.cfg.reward.crop_cost * item.crops
                    for item in summaries
                ]
            )
        )
        print(
            f"[eval] images={len(summaries)} score={score:.4f} "
            f"reward={np.mean([item.reward for item in summaries]):.3f} "
            f"crops={np.mean([item.crops for item in summaries]):.2f}"
        )
        return score

    def _run_episode(
        self,
        image_path: Path,
        split: str,
        epsilon: float,
        learn: bool,
    ) -> EpisodeSummary:
        image = read_image(image_path)
        image_shape = image.shape[:2]
        label_path = label_path_for(image_path, self.cfg.paths.label_root, split)
        ground_truth = load_yolo_labels(label_path, image_shape).filter_classes(
            self.cfg.detector.target_classes
        )
        full, full_ms, _ = self.runner.full(
            image,
            image_path,
            split,
            use_cache=self.cfg.train.cache_full_detections,
        )
        hard_regions, _ = self.runner.hard_regions(
            full,
            ground_truth,
            image_path,
            label_path,
            split,
            use_cache=self.cfg.train.cache_hard_regions,
        )
        hard_mask = hard_regions.hard_mask
        anchors = generate_anchors(full, image_shape, self.cfg.anchors)
        environment = AnchorZoomEnvironment(
            anchors, full, image_shape, self.cfg.anchors, self.cfg.environment
        )
        predictions = full.filter_score(self.cfg.detector.output_confidence)
        initial_stats = match_stats(
            predictions,
            ground_truth,
            image_shape,
            self.cfg.reward.match_iou,
            self.cfg.reward.small_area_ratio,
            hard_mask,
        )
        n_step = NStepAccumulator(self.cfg.train.n_step, self.cfg.train.gamma)
        total_reward = 0.0
        accepted_count = 0
        crop_ms = 0.0
        losses: list[float] = []
        stopped = False

        while True:
            state = environment.state()
            action_mask = environment.action_mask()
            action = self.agent.select_action(state, action_mask, epsilon)
            if learn:
                self.environment_steps += 1

            if action == environment.stop_action:
                stopped = True
                reward = self._stop_reward(
                    predictions,
                    ground_truth,
                    hard_mask,
                    environment,
                    action_mask,
                )
                next_state = environment.state()
                next_mask = np.zeros_like(action_mask)
                next_mask[environment.stop_action] = True
                transition = Transition(
                    state, action, reward, next_state, True, next_mask
                )
                total_reward += reward
                if learn:
                    losses.extend(self._store_and_optimize(n_step, transition))
                break

            roi = environment.roi_for_action(action)
            overlap = environment.overlap_with_history(roi)
            crop, elapsed_ms, _ = self.runner.crop(
                image,
                image_path,
                split,
                roi,
                use_cache=self.cfg.train.cache_crop_detections,
            )
            crop_ms += elapsed_ms
            crop = crop.filter_score(self.cfg.detector.output_confidence)
            before = match_stats(
                predictions,
                ground_truth,
                image_shape,
                self.cfg.reward.match_iou,
                self.cfg.reward.small_area_ratio,
                hard_mask,
            )
            utility = crop_utility(crop, predictions, self.cfg.detector.duplicate_iou)
            eligible = (
                len(crop) >= self.cfg.reward.min_crop_detections
                and utility >= self.cfg.reward.min_utility
            )
            candidate = (
                merge_detections(
                    predictions,
                    crop,
                    self.cfg.detector.merge_iou,
                    self.cfg.detector.max_detections,
                )
                if eligible
                else predictions
            )
            after = match_stats(
                candidate,
                ground_truth,
                image_shape,
                self.cfg.reward.match_iou,
                self.cfg.reward.small_area_ratio,
                hard_mask,
            )
            outcome = crop_step_outcome(
                before, after, utility, overlap, len(crop), self.cfg.reward
            )
            if outcome.accepted:
                predictions = candidate
                accepted_count += 1
            environment.record(action, roi, outcome.accepted, outcome.utility)
            total_reward += outcome.reward

            next_state = environment.state()
            next_mask = environment.action_mask()
            done = environment.done or int(next_mask.sum()) == 1
            transition = Transition(
                state,
                action,
                outcome.reward,
                next_state,
                done,
                next_mask,
            )
            if learn:
                losses.extend(self._store_and_optimize(n_step, transition))
            if done:
                break

        final_stats = match_stats(
            predictions,
            ground_truth,
            image_shape,
            self.cfg.reward.match_iou,
            self.cfg.reward.small_area_ratio,
            hard_mask,
        )
        return EpisodeSummary(
            reward=float(total_reward),
            crops=len(environment.attempted_rois),
            accepted=accepted_count,
            tp_gain=final_stats.true_positives - initial_stats.true_positives,
            hard_tp_gain=(
                final_stats.hard_true_positives
                - initial_stats.hard_true_positives
            ),
            small_tp_gain=(
                final_stats.small_true_positives - initial_stats.small_true_positives
            ),
            stopped=stopped,
            loss=float(np.mean(losses)) if losses else None,
            full_ms=float(full_ms),
            crop_ms=float(crop_ms),
        )

    def _store_and_optimize(
        self,
        accumulator: NStepAccumulator,
        transition: Transition,
    ) -> list[float]:
        losses: list[float] = []
        for aggregated in accumulator.append(transition):
            self.replay.add(aggregated)
            loss = self.agent.optimize(self.replay)
            if loss is not None:
                losses.append(loss)
        return losses

    def _stop_reward(
        self,
        predictions: Detections,
        ground_truth: Detections,
        hard_mask: np.ndarray,
        environment: AnchorZoomEnvironment,
        action_mask: np.ndarray,
    ) -> float:
        stats = match_stats(
            predictions,
            ground_truth,
            environment.image_shape,
            self.cfg.reward.match_iou,
            self.cfg.reward.small_area_ratio,
            hard_mask,
        )
        unmatched_indices = np.flatnonzero(~stats.matched_gt)
        unmatched = ground_truth.boxes[unmatched_indices]
        if len(unmatched) == 0:
            return float(self.cfg.reward.stop_bonus)
        centers = (unmatched[:, :2] + unmatched[:, 2:]) * 0.5
        unmatched_hard = hard_mask[unmatched_indices]
        useful_regular_remaining = False
        for action in np.flatnonzero(action_mask):
            if action == environment.stop_action:
                continue
            roi = environment.roi_for_action(int(action))
            inside = (
                (centers[:, 0] >= roi[0])
                & (centers[:, 0] <= roi[2])
                & (centers[:, 1] >= roi[1])
                & (centers[:, 1] <= roi[3])
            )
            if bool((inside & unmatched_hard).any()):
                return -float(self.cfg.reward.stop_hard_early_penalty)
            useful_regular_remaining = useful_regular_remaining or bool(inside.any())
        if useful_regular_remaining:
            return -float(self.cfg.reward.stop_early_penalty)
        return float(self.cfg.reward.stop_bonus)

    def _resume(self) -> None:
        if not self.latest_checkpoint.exists():
            print(f"[train] resume requested but checkpoint is missing: {self.latest_checkpoint}")
            return
        payload = self.agent.load(self.latest_checkpoint, replay=self.replay)
        self.start_episode = int(payload.get("episode", 0)) + 1
        self.environment_steps = int(payload.get("environment_steps", 0))
        self.best_score = float(payload.get("best_score", float("-inf")))
        print(
            f"[train] resumed episode={self.start_episode} "
            f"steps={self.environment_steps} replay={len(self.replay)}"
        )

    def _prepare_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self.output_dir / "config.snapshot.json"
        snapshot.write_text(
            json.dumps(self.cfg.as_dict(), indent=2, default=str, sort_keys=True),
            encoding="utf-8",
        )
        if not self.log_path.exists():
            self.log_path.write_text(
                "episode,image,epsilon,reward,crops,accepted,tp_gain,hard_tp_gain,"
                "small_tp_gain,"
                "stopped,loss,full_ms,crop_ms\n",
                encoding="utf-8",
            )

    def _append_log(
        self,
        episode: int,
        image_path: Path,
        epsilon: float,
        summary: EpisodeSummary,
    ) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{episode},{image_path.name},{epsilon:.6f},{summary.reward:.6f},"
                f"{summary.crops},{summary.accepted},{summary.tp_gain},"
                f"{summary.hard_tp_gain},{summary.small_tp_gain},"
                f"{int(summary.stopped)},"
                f"{'' if summary.loss is None else f'{summary.loss:.6f}'},"
                f"{summary.full_ms:.3f},{summary.crop_ms:.3f}\n"
            )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_loss(items: list[EpisodeSummary]) -> float:
    values = [item.loss for item in items if item.loss is not None]
    return float(np.mean(values)) if values else 0.0
