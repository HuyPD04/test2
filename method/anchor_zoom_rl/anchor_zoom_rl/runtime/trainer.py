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
from ..core.postprocess import crop_reliability, crop_utility, merge_detections
from ..core.reward import (
    build_hard_action_supervision,
    covered_ground_truth_mask,
    crop_step_outcome,
    match_stats,
    stop_reward,
)
from ..core.state import state_dimension
from ..core.types import Detections
from ..rl.agent import DQNAgent
from ..rl.replay import NStepAccumulator, ReplayBuffer, Transition
from .data import (
    iter_images,
    label_path_for,
    load_yolo_labels,
    read_image,
    stratified_sequence_sample,
)
from .environment import AnchorZoomEnvironment
from .metrics import AP50Accumulator
from .prediction import DetectionRunner


TRAINING_SCHEMA_VERSION = 3


@dataclass(slots=True)
class EpisodeSummary:
    reward: float
    crops: int
    accepted: int
    tp_gain: int
    hard_tp_gain: int
    small_tp_gain: int
    hard_coverage_gain: int
    hard_gt: int
    hard_true_positives: int
    attempted_hard_coverage: int
    accepted_hard_coverage: int
    stopped: bool
    loss: float | None
    td_loss: float | None
    hard_aux_loss: float | None
    full_ms: float
    crop_ms: float
    method_ms: float
    predictions: Detections
    ground_truth: Detections


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
        self.val_images = stratified_sequence_sample(
            iter_images(cfg.paths.image_root, val_split),
            cfg.train.eval_images,
            cfg.train.seed,
        )
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
        self.best_ap = float("-inf")
        self.best_hard_recall = float("-inf")
        self.rng = random.Random(cfg.train.seed)
        self._sampling_epoch = -1
        self._sampling_order: list[Path] = []
        self.output_dir = cfg.paths.output_dir
        self.latest_checkpoint = self.output_dir / "checkpoints" / "latest.pt"
        self.best_tradeoff_checkpoint = (
            self.output_dir / "checkpoints" / "best_tradeoff.pt"
        )
        self.best_ap_checkpoint = self.output_dir / "checkpoints" / "best_ap.pt"
        self.best_hard_checkpoint = (
            self.output_dir / "checkpoints" / "best_hard_recall.pt"
        )
        self.log_path = self.output_dir / "train.csv"
        self.eval_log_path = self.output_dir / "eval.csv"
        _set_seed(cfg.train.seed)
        if (
            not cfg.train.resume
            and (self.log_path.exists() or self.latest_checkpoint.exists())
        ):
            raise FileExistsError(
                f"Fresh training output already exists: {self.output_dir}. "
                "Use a new --out-dir or pass --resume for the same run."
            )
        self._prepare_output()
        if cfg.train.resume:
            self._resume()

    def train(self, episodes: int | None = None) -> Path:
        total_episodes = int(episodes or self.cfg.train.episodes)
        window: list[EpisodeSummary] = []
        for episode in range(self.start_episode, total_episodes + 1):
            image_path = self._training_image(episode)
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
                    f"loss={_mean_loss(recent):.4f} "
                    f"hard_aux={_mean_hard_aux_loss(recent):.4f}"
                )

            if episode % max(self.cfg.train.eval_interval, 1) == 0 and self.val_images:
                evaluation = self.evaluate(episode)
                self._save_validation_checkpoints(episode, evaluation)

            if episode % max(self.cfg.train.checkpoint_interval, 1) == 0:
                self.agent.save(
                    self.latest_checkpoint,
                    episode,
                    self.environment_steps,
                    replay=self.replay,
                    best_score=self.best_score,
                    best_metrics=self._best_metrics(),
                    training_schema_version=TRAINING_SCHEMA_VERSION,
                )

        final_episode = max(total_episodes, self.start_episode - 1)
        self.agent.save(
            self.latest_checkpoint,
            final_episode,
            self.environment_steps,
            replay=self.replay,
            best_score=self.best_score,
            best_metrics=self._best_metrics(),
            training_schema_version=TRAINING_SCHEMA_VERSION,
        )
        if not self.cfg.paths.checkpoint.exists():
            self.agent.save(
                self.cfg.paths.checkpoint,
                final_episode,
                self.environment_steps,
                best_score=self.best_score,
                best_metrics=self._best_metrics(),
                training_schema_version=TRAINING_SCHEMA_VERSION,
            )
        elif not self.cfg.train.resume and self.best_score == float("-inf"):
            self.agent.save(
                self.cfg.paths.checkpoint,
                final_episode,
                self.environment_steps,
                best_score=self.best_score,
                best_metrics=self._best_metrics(),
                training_schema_version=TRAINING_SCHEMA_VERSION,
            )
        return self.cfg.paths.checkpoint

    def _training_image(self, episode: int) -> Path:
        if self.cfg.train.sampling_mode == "random_with_replacement":
            return self.rng.choice(self.images)
        if self.cfg.train.sampling_mode != "shuffled_epochs":
            raise ValueError(
                "train.sampling_mode must be 'shuffled_epochs' or "
                "'random_with_replacement'"
            )
        epoch = (int(episode) - 1) // len(self.images)
        offset = (int(episode) - 1) % len(self.images)
        if epoch != self._sampling_epoch:
            self._sampling_order = list(self.images)
            random.Random(self.cfg.train.seed + epoch).shuffle(self._sampling_order)
            self._sampling_epoch = epoch
        return self._sampling_order[offset]

    def evaluate(self, episode: int) -> dict[str, float]:
        summaries = [
            self._run_episode(path, self.val_split, epsilon=0.0, learn=False)
            for path in self.val_images
        ]
        metric = AP50Accumulator(
            self.cfg.detector.target_classes,
            self.cfg.reward.match_iou,
        )
        for item in summaries:
            metric.update(item.predictions, item.ground_truth)
        quality = metric.compute()
        fp_per_image = quality["false_positives"] / max(len(summaries), 1)
        mean_crops = float(np.mean([item.crops for item in summaries]))
        mean_reward = float(np.mean([item.reward for item in summaries]))
        mean_method_ms = float(np.mean([item.method_ms for item in summaries]))
        hard_gt = int(sum(item.hard_gt for item in summaries))
        hard_true_positives = int(
            sum(item.hard_true_positives for item in summaries)
        )
        attempted_hard_coverage = int(
            sum(item.attempted_hard_coverage for item in summaries)
        )
        accepted_hard_coverage = int(
            sum(item.accepted_hard_coverage for item in summaries)
        )
        hard_recall = hard_true_positives / max(hard_gt, 1)
        attempted_hard_coverage_rate = attempted_hard_coverage / max(hard_gt, 1)
        accepted_hard_coverage_rate = accepted_hard_coverage / max(hard_gt, 1)
        score = float(
            self.cfg.train.eval_ap_weight * quality["ap50"]
            + self.cfg.train.eval_hard_recall_weight * hard_recall
            - self.cfg.train.eval_fp_per_image_weight * fp_per_image
            - self.cfg.train.eval_crop_weight * mean_crops
        )
        evaluation = {
            "score": score,
            "ap50": float(quality["ap50"]),
            "hard_recall": float(hard_recall),
            "hard_gt": float(hard_gt),
            "hard_true_positives": float(hard_true_positives),
            "attempted_hard_coverage": float(attempted_hard_coverage_rate),
            "accepted_hard_coverage": float(accepted_hard_coverage_rate),
            "method_latency_ms": float(mean_method_ms),
        }
        self._append_eval_log(
            episode,
            score,
            quality,
            fp_per_image,
            mean_crops,
            mean_reward,
            evaluation,
            len(summaries),
        )
        print(
            f"[eval] images={len(summaries)} score={score:.4f} "
            f"ap50={quality['ap50']:.4f} precision={quality['precision']:.4f} "
            f"recall={quality['recall']:.4f} fp/image={fp_per_image:.2f} "
            f"hard_recall={hard_recall:.4f} "
            f"hard_coverage={attempted_hard_coverage_rate:.4f}/"
            f"{accepted_hard_coverage_rate:.4f} crops={mean_crops:.2f}"
        )
        return evaluation

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
        method_start = time.perf_counter()
        anchors = generate_anchors(full, image_shape, self.cfg.anchors)
        environment = AnchorZoomEnvironment(
            anchors, full, image_shape, self.cfg.anchors, self.cfg.environment
        )
        predictions = merge_detections(
            Detections.empty(),
            full.filter_score(self.cfg.detector.output_confidence),
            self.cfg.detector.merge_iou,
            self.cfg.detector.max_detections,
            self.cfg.detector.cross_class_iou,
            self.cfg.detector.cross_class_ios,
            self.cfg.detector.cross_class_score_ratio,
            self.cfg.detector.cross_class_groups,
        )
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
        hard_coverage_gain = 0
        crop_ms = 0.0
        losses: list[float] = []
        td_losses: list[float] = []
        hard_aux_losses: list[float] = []
        stopped = False

        while True:
            state = environment.state()
            action_mask = environment.action_mask()
            before = match_stats(
                predictions,
                ground_truth,
                image_shape,
                self.cfg.reward.match_iou,
                self.cfg.reward.small_area_ratio,
                hard_mask,
            )
            supervision = self._hard_action_supervision(
                ground_truth,
                before.matched_gt,
                hard_mask,
                environment,
                action_mask,
            )
            action = self.agent.select_action(state, action_mask, epsilon)
            if learn:
                self.environment_steps += 1

            if action == environment.stop_action:
                stopped = True
                reward = stop_reward(
                    supervision.reachable_hard_count,
                    supervision.reachable_regular_count,
                    self.cfg.reward,
                )
                next_state = environment.state()
                next_mask = np.zeros_like(action_mask)
                next_mask[environment.stop_action] = True
                transition = Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    True,
                    next_mask,
                    hard_targets=supervision.targets,
                    hard_target_mask=supervision.target_mask,
                )
                total_reward += reward
                if learn:
                    updates = self._store_and_optimize(n_step, transition)
                    losses.extend(item[0] for item in updates)
                    td_losses.extend(item[1] for item in updates)
                    hard_aux_losses.extend(item[2] for item in updates)
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
            utility = crop_utility(
                crop,
                predictions,
                self.cfg.detector.duplicate_iou,
                self.cfg.reward.refinement_iou,
                self.cfg.reward.refinement_score_ratio,
                self.cfg.reward.refinement_utility_weight,
            )
            anchor_index, _ = environment.decode_action(action)
            reliability = crop_reliability(
                crop,
                predictions,
                roi,
                anchor_score=environment.anchors[anchor_index].score,
                history_overlap=overlap,
                duplicate_iou=self.cfg.detector.duplicate_iou,
                refinement_iou=self.cfg.reward.refinement_iou,
                refinement_score_ratio=self.cfg.reward.refinement_score_ratio,
            )
            eligible = (
                len(crop) >= self.cfg.reward.min_crop_detections
                and utility >= self.cfg.reward.min_utility
                and reliability >= self.cfg.reward.min_reliability
            )
            candidate = (
                merge_detections(
                    predictions,
                    crop,
                    self.cfg.detector.merge_iou,
                    self.cfg.detector.max_detections,
                    self.cfg.detector.cross_class_iou,
                    self.cfg.detector.cross_class_ios,
                    self.cfg.detector.cross_class_score_ratio,
                    self.cfg.detector.cross_class_groups,
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
                before,
                after,
                utility,
                overlap,
                len(crop),
                self.cfg.reward,
                reliability,
                int(supervision.newly_covered_counts[action]),
            )
            if outcome.accepted:
                predictions = candidate
                accepted_count += 1
            environment.record(action, roi, outcome.accepted, outcome.utility)
            total_reward += outcome.reward
            hard_coverage_gain += outcome.hard_coverage_gain

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
                hard_targets=supervision.targets,
                hard_target_mask=supervision.target_mask,
            )
            if learn:
                updates = self._store_and_optimize(n_step, transition)
                losses.extend(item[0] for item in updates)
                td_losses.extend(item[1] for item in updates)
                hard_aux_losses.extend(item[2] for item in updates)
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
        attempted_hard = (
            covered_ground_truth_mask(ground_truth, environment.attempted_rois)
            & hard_mask
        )
        accepted_hard = (
            covered_ground_truth_mask(ground_truth, environment.accepted_rois)
            & hard_mask
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
            hard_coverage_gain=int(hard_coverage_gain),
            hard_gt=int(hard_mask.sum()),
            hard_true_positives=int(final_stats.hard_true_positives),
            attempted_hard_coverage=int(attempted_hard.sum()),
            accepted_hard_coverage=int(accepted_hard.sum()),
            stopped=stopped,
            loss=float(np.mean(losses)) if losses else None,
            td_loss=float(np.mean(td_losses)) if td_losses else None,
            hard_aux_loss=(
                float(np.mean(hard_aux_losses)) if hard_aux_losses else None
            ),
            full_ms=float(full_ms),
            crop_ms=float(crop_ms),
            method_ms=float((time.perf_counter() - method_start) * 1000.0),
            predictions=predictions,
            ground_truth=ground_truth,
        )

    def _store_and_optimize(
        self,
        accumulator: NStepAccumulator,
        transition: Transition,
    ) -> list[tuple[float, float, float]]:
        losses: list[tuple[float, float, float]] = []
        for aggregated in accumulator.append(transition):
            clip = float(self.cfg.train.reward_clip)
            if clip > 0:
                aggregated.reward = float(
                    np.clip(aggregated.reward, -clip, clip)
                )
            self.replay.add(aggregated)
            loss = self.agent.optimize(self.replay)
            if loss is not None:
                losses.append(
                    (
                        loss,
                        self.agent.last_td_loss,
                        self.agent.last_hard_aux_loss,
                    )
                )
        return losses

    def _hard_action_supervision(
        self,
        ground_truth: Detections,
        matched_gt: np.ndarray,
        hard_mask: np.ndarray,
        environment: AnchorZoomEnvironment,
        action_mask: np.ndarray,
    ):
        target_mask = np.asarray(action_mask, dtype=bool).copy()
        target_mask[environment.stop_action] = False
        action_rois = np.zeros((environment.action_count, 4), dtype=np.float32)
        for action in np.flatnonzero(target_mask):
            action_rois[action] = environment.roi_for_action(int(action))
        return build_hard_action_supervision(
            ground_truth,
            matched_gt,
            hard_mask,
            action_rois,
            target_mask,
            environment.attempted_rois,
        )

    def _save_validation_checkpoints(
        self,
        episode: int,
        evaluation: dict[str, float],
    ) -> None:
        improved_tradeoff = evaluation["score"] > self.best_score
        improved_ap = evaluation["ap50"] > self.best_ap
        improved_hard = evaluation["hard_recall"] > self.best_hard_recall
        if improved_tradeoff:
            self.best_score = evaluation["score"]
        if improved_ap:
            self.best_ap = evaluation["ap50"]
        if improved_hard:
            self.best_hard_recall = evaluation["hard_recall"]

        paths: list[tuple[Path, str, float]] = []
        if improved_tradeoff:
            paths.append(
                (self.best_tradeoff_checkpoint, "tradeoff", evaluation["score"])
            )
            paths.append((self.cfg.paths.checkpoint, "best alias", evaluation["score"]))
        if improved_ap:
            paths.append((self.best_ap_checkpoint, "AP50", evaluation["ap50"]))
        if improved_hard:
            paths.append(
                (
                    self.best_hard_checkpoint,
                    "hard recall",
                    evaluation["hard_recall"],
                )
            )
        saved: set[Path] = set()
        for path, label, value in paths:
            resolved = path.resolve()
            if resolved in saved:
                continue
            self.agent.save(
                path,
                episode,
                self.environment_steps,
                best_score=self.best_score,
                best_metrics=self._best_metrics(),
                training_schema_version=TRAINING_SCHEMA_VERSION,
            )
            saved.add(resolved)
            print(f"[train] new best {label}={value:.4f} checkpoint={path.name}")

    def _best_metrics(self) -> dict[str, float]:
        return {
            "tradeoff": float(self.best_score),
            "ap50": float(self.best_ap),
            "hard_recall": float(self.best_hard_recall),
        }

    def _resume(self) -> None:
        if not self.latest_checkpoint.exists():
            print(f"[train] resume requested but checkpoint is missing: {self.latest_checkpoint}")
            return
        try:
            header = torch.load(
                self.latest_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            header = torch.load(self.latest_checkpoint, map_location="cpu")
        schema_version = int(header.get("training_schema_version", 1))
        if schema_version != TRAINING_SCHEMA_VERSION:
            raise ValueError(
                "Checkpoint uses an incompatible training reward/evaluation "
                f"schema ({schema_version} != {TRAINING_SCHEMA_VERSION}). "
                "Start a fresh run with a new --out-dir; the checkpoint remains "
                "valid for inference."
            )
        payload = self.agent.load(self.latest_checkpoint, replay=self.replay)
        self.start_episode = int(payload.get("episode", 0)) + 1
        self.environment_steps = int(payload.get("environment_steps", 0))
        self.best_score = float(payload.get("best_score", float("-inf")))
        best_metrics = payload.get("best_metrics", {})
        self.best_ap = float(best_metrics.get("ap50", float("-inf")))
        self.best_hard_recall = float(
            best_metrics.get("hard_recall", float("-inf"))
        )
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
                "small_tp_gain,hard_coverage_gain,"
                "stopped,loss,td_loss,hard_aux_loss,full_ms,crop_ms\n",
                encoding="utf-8",
            )
        if not self.eval_log_path.exists():
            self.eval_log_path.write_text(
                "episode,score,ap50,precision,recall,false_positives,"
                "fp_per_image,mean_crops,mean_reward,hard_recall,hard_tp,hard_gt,"
                "attempted_hard_coverage,accepted_hard_coverage,"
                "method_latency_ms,images\n",
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
                f"{summary.hard_coverage_gain},"
                f"{int(summary.stopped)},"
                f"{'' if summary.loss is None else f'{summary.loss:.6f}'},"
                f"{'' if summary.td_loss is None else f'{summary.td_loss:.6f}'},"
                f"{'' if summary.hard_aux_loss is None else f'{summary.hard_aux_loss:.6f}'},"
                f"{summary.full_ms:.3f},{summary.crop_ms:.3f}\n"
            )

    def _append_eval_log(
        self,
        episode: int,
        score: float,
        quality: dict,
        fp_per_image: float,
        mean_crops: float,
        mean_reward: float,
        evaluation: dict[str, float],
        images: int,
    ) -> None:
        with self.eval_log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{episode},{score:.6f},{quality['ap50']:.6f},"
                f"{quality['precision']:.6f},{quality['recall']:.6f},"
                f"{quality['false_positives']},{fp_per_image:.6f},"
                f"{mean_crops:.6f},{mean_reward:.6f},"
                f"{evaluation['hard_recall']:.6f},"
                f"{int(evaluation['hard_true_positives'])},"
                f"{int(evaluation['hard_gt'])},"
                f"{evaluation['attempted_hard_coverage']:.6f},"
                f"{evaluation['accepted_hard_coverage']:.6f},"
                f"{evaluation['method_latency_ms']:.6f},"
                f"{images}\n"
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


def _mean_hard_aux_loss(items: list[EpisodeSummary]) -> float:
    values = [item.hard_aux_loss for item in items if item.hard_aux_loss is not None]
    return float(np.mean(values)) if values else 0.0
