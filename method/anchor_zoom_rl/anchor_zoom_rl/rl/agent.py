from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from ..config import TrainConfig
from .network import DuelingQNetwork
from .replay import ReplayBuffer


def resolve_torch_device(requested: str) -> torch.device:
    name = str(requested or "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA is unavailable; falling back to CPU.")
        name = "cpu"
    return torch.device(name)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: TrainConfig,
        device: str,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg
        self.device = resolve_torch_device(device)
        self.online = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.optimization_steps = 0
        self.last_td_loss = 0.0
        self.last_hard_aux_loss = 0.0

    def epsilon(self, environment_steps: int) -> float:
        progress = min(max(environment_steps, 0) / max(self.cfg.epsilon_decay_steps, 1), 1.0)
        return float(
            self.cfg.epsilon_start
            + progress * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        )

    def select_action(
        self,
        state: np.ndarray,
        valid_mask: np.ndarray,
        epsilon: float = 0.0,
    ) -> int:
        valid = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
        if len(valid) == 0:
            raise ValueError("At least one action must be valid")
        if random.random() < float(epsilon):
            return int(random.choice(valid.tolist()))
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(tensor)[0]
            mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min)
            return int(torch.argmax(q_values).item())

    def select_action_with_hardness(
        self,
        state: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        with torch.no_grad():
            tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values, hard_logits = self.online.forward_with_hardness(tensor)
            mask = torch.as_tensor(
                valid_mask, dtype=torch.bool, device=self.device
            )
            masked_q = q_values[0].masked_fill(
                ~mask, torch.finfo(q_values.dtype).min
            )
            action = int(torch.argmax(masked_q).item())
            probabilities = (
                torch.sigmoid(hard_logits[0]).cpu().numpy().astype(np.float32)
            )
            return action, probabilities

    def optimize(self, replay: ReplayBuffer) -> float | None:
        if len(replay) < max(self.cfg.min_replay, self.cfg.batch_size):
            return None
        batch = replay.sample(self.cfg.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_masks = torch.as_tensor(
            np.stack([item.next_mask for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        discounts = torch.as_tensor(
            [item.discount for item in batch], dtype=torch.float32, device=self.device
        )

        q_values, hard_logits = self.online.forward_with_hardness(states)
        predicted = q_values.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            online_next = self.online(next_states).masked_fill(
                ~next_masks, torch.finfo(torch.float32).min
            )
            next_actions = online_next.argmax(dim=1)
            target_next = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
            expected = rewards + discounts * target_next
        td_loss = F.smooth_l1_loss(predicted, expected)
        hard_targets, hard_target_mask = self._hard_target_batch(batch)
        if bool(hard_target_mask.any()):
            selected_logits = hard_logits[hard_target_mask]
            selected_targets = hard_targets[hard_target_mask]
            positives = selected_targets.sum()
            negatives = selected_targets.numel() - positives
            positive_weight = torch.clamp(
                negatives / torch.clamp(positives, min=1.0),
                min=1.0,
                max=float(self.cfg.hard_aux_positive_weight_max),
            )
            hard_aux_loss = F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_targets,
                pos_weight=positive_weight,
            )
        else:
            hard_aux_loss = torch.zeros((), dtype=td_loss.dtype, device=self.device)
        loss = td_loss + float(self.cfg.hard_aux_loss_weight) * hard_aux_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()
        self.optimization_steps += 1
        self.last_td_loss = float(td_loss.detach().cpu().item())
        self.last_hard_aux_loss = float(hard_aux_loss.detach().cpu().item())
        self._update_target()
        return float(loss.detach().cpu().item())

    def _hard_target_batch(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = np.zeros((len(batch), self.action_dim), dtype=np.float32)
        masks = np.zeros((len(batch), self.action_dim), dtype=bool)
        for index, item in enumerate(batch):
            if item.hard_targets is None or item.hard_target_mask is None:
                continue
            values = np.asarray(item.hard_targets, dtype=np.float32).reshape(-1)
            valid = np.asarray(item.hard_target_mask, dtype=bool).reshape(-1)
            if len(values) != self.action_dim or len(valid) != self.action_dim:
                raise ValueError("Hard-action targets must match the action dimension")
            targets[index] = values
            masks[index] = valid
        return (
            torch.as_tensor(targets, dtype=torch.float32, device=self.device),
            torch.as_tensor(masks, dtype=torch.bool, device=self.device),
        )

    def _update_target(self) -> None:
        tau = float(self.cfg.soft_update_tau)
        if tau > 0:
            with torch.no_grad():
                for target_param, online_param in zip(
                    self.target.parameters(), self.online.parameters(), strict=True
                ):
                    target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)
        elif self.optimization_steps % max(self.cfg.target_update_interval, 1) == 0:
            self.target.load_state_dict(self.online.state_dict())

    def save(
        self,
        path: Path,
        episode: int,
        environment_steps: int,
        replay: ReplayBuffer | None = None,
        best_score: float | None = None,
        best_metrics: dict[str, float] | None = None,
        training_schema_version: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "episode": int(episode),
            "environment_steps": int(environment_steps),
            "optimization_steps": int(self.optimization_steps),
            "best_score": best_score,
            "best_metrics": dict(best_metrics or {}),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if training_schema_version is not None:
            payload["training_schema_version"] = int(training_schema_version)
        if replay is not None:
            payload["replay"] = replay.state_dict()
        torch.save(payload, path)

    def load(
        self,
        path: Path,
        replay: ReplayBuffer | None = None,
        load_optimizer: bool = True,
    ) -> dict:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        if int(payload["state_dim"]) != self.state_dim or int(payload["action_dim"]) != self.action_dim:
            raise ValueError(
                "Checkpoint action/state dimensions do not match the current anchor configuration"
            )
        self._load_network_state(self.online, payload["online"])
        self._load_network_state(
            self.target, payload.get("target", payload["online"])
        )
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.optimization_steps = int(payload.get("optimization_steps", 0))
        if replay is not None and "replay" in payload:
            replay.load_state_dict(payload["replay"])
        return payload

    @staticmethod
    def _load_network_state(network: DuelingQNetwork, state: dict) -> None:
        incompatible = network.load_state_dict(state, strict=False)
        invalid_missing = [
            key for key in incompatible.missing_keys if not key.startswith("hardness.")
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise ValueError(
                "Checkpoint network structure is incompatible: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
