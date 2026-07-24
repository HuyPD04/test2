from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray
    discount: float = 1.0


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.data: list[Transition] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.data, int(batch_size))

    def state_dict(self) -> dict:
        return {"data": self.data, "position": self.position, "capacity": self.capacity}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.data = list(state.get("data", []))[-self.capacity :]
        self.position = int(state.get("position", len(self.data) % max(self.capacity, 1)))


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = max(int(n_step), 1)
        self.gamma = float(gamma)
        self.pending: deque[Transition] = deque()

    def append(self, transition: Transition) -> list[Transition]:
        self.pending.append(transition)
        ready: list[Transition] = []
        if len(self.pending) >= self.n_step:
            ready.append(self._aggregate())
            self.pending.popleft()
        if transition.done:
            while self.pending:
                ready.append(self._aggregate())
                self.pending.popleft()
        return ready

    def _aggregate(self) -> Transition:
        reward = 0.0
        steps = 0
        final = self.pending[0]
        for item in list(self.pending)[: self.n_step]:
            reward += (self.gamma**steps) * float(item.reward)
            steps += 1
            final = item
            if item.done:
                break
        first = self.pending[0]
        return Transition(
            state=first.state,
            action=first.action,
            reward=float(reward),
            next_state=final.next_state,
            done=final.done,
            next_mask=final.next_mask,
            discount=0.0 if final.done else float(self.gamma**steps),
        )
