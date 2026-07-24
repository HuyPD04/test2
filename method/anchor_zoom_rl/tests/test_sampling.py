from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from anchor_zoom_rl.runtime.data import stratified_sequence_sample
from anchor_zoom_rl.runtime.trainer import AnchorZoomTrainer


def test_stratified_sample_covers_sequences_before_repeating() -> None:
    images = [
        Path(f"0001_{index}.jpg") for index in range(5)
    ] + [
        Path(f"0002_{index}.jpg") for index in range(5)
    ]
    sampled = stratified_sequence_sample(images, limit=4, seed=42)
    prefixes = [path.stem.split("_", 1)[0] for path in sampled]
    assert prefixes.count("0001") == 2
    assert prefixes.count("0002") == 2
    assert len(sampled) == len(set(sampled)) == 4


def test_shuffled_epoch_sampler_visits_every_image_once() -> None:
    trainer = AnchorZoomTrainer.__new__(AnchorZoomTrainer)
    trainer.images = [Path(f"{index}.jpg") for index in range(8)]
    trainer.cfg = SimpleNamespace(
        train=SimpleNamespace(sampling_mode="shuffled_epochs", seed=7)
    )
    trainer.rng = __import__("random").Random(7)
    trainer._sampling_epoch = -1
    trainer._sampling_order = []

    first_epoch = [trainer._training_image(episode) for episode in range(1, 9)]
    second_epoch = [trainer._training_image(episode) for episode in range(9, 17)]
    assert len(set(first_epoch)) == 8
    assert len(set(second_epoch)) == 8
    assert first_epoch != second_epoch
