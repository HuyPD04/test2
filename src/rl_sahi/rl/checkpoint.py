from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch

from rl_sahi.common.actions import ACTION_NAMES, NUM_ACTIONS
from rl_sahi.common.device import DeviceLike, resolve_torch_device
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.state_config import StateConfig, StateLayout


def save_checkpoint(
    path: Path,
    policy: QNetwork,
    state_dim: int,
    train_cfg: Any,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    layout: StateLayout | None = None,
    detection_metadata: dict[str, Any] | None = None,
    training_metadata: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "model": policy.state_dict(),
            "state_dim": state_dim,
            "network_type": "spatial_cnn" if policy.use_spatial_cnn else "mlp",
            "dueling": policy.dueling,
            "state_layout": asdict(layout) if layout is not None else None,
            "train_cfg": asdict(train_cfg),
            "env_cfg": asdict(env_cfg),
            "state_cfg": asdict(state_cfg),
            "detection_metadata": detection_metadata,
            "training_metadata": training_metadata or {},
            "actions": {int(k): v for k, v in ACTION_NAMES.items()},
        },
        path,
    )


def build_training_metadata(
    target_classes: tuple[int, ...],
    class_mapping: Any,
    infer_cfg: Any,
    bench_cfg: Any,
) -> dict[str, Any]:
    return {
        "target_classes": tuple(int(x) for x in target_classes),
        "class_mapping": {
            "model_to_label": dict(getattr(class_mapping, "model_to_label", {}) or {}),
            "label_to_eval": dict(getattr(class_mapping, "label_to_eval", {}) or {}),
        },
        "inference_config": asdict(infer_cfg) if is_dataclass(infer_cfg) else {},
        "benchmark_config": asdict(bench_cfg) if is_dataclass(bench_cfg) else {},
    }


_DISABLE_ONNX_AUTOLOAD = False

class ONNXPolicyWrapper:
    def __init__(self, onnx_path: str, device: DeviceLike = None):
        import onnxruntime as ort
        device_str = str(device).lower() if device is not None else "cpu"
        providers = ['CUDAExecutionProvider'] if "cuda" in device_str else ['CPUExecutionProvider']
        
        # fallback to CPU if CUDA is requested but not available in ORT
        if "cuda" in device_str and 'CUDAExecutionProvider' not in ort.get_available_providers():
            providers = ['CPUExecutionProvider']
            
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # Expose input_dim for compatibility with rollout.py
        shape = self.session.get_inputs()[0].shape
        self.input_dim = shape[1] if len(shape) > 1 else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        inputs = {self.input_name: x.cpu().numpy()}
        outs = self.session.run(None, inputs)
        return torch.from_numpy(outs[0]).to(x.device)

def load_policy(checkpoint_path: Path, device: DeviceLike = None) -> tuple[Any, dict]:
    device = resolve_torch_device(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (TypeError, Exception):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    env_allowed = {field.name for field in fields(EnvConfig)}
    state_allowed = {field.name for field in fields(StateConfig)}
    env_cfg = EnvConfig(**{key: value for key, value in checkpoint.get("env_cfg", {}).items() if key in env_allowed})
    state_cfg = StateConfig(**{key: value for key, value in checkpoint.get("state_cfg", {}).items() if key in state_allowed})
    hidden_dim = checkpoint.get("train_cfg", {}).get("hidden_dim", 512)
    layout_data = checkpoint.get("state_layout")
    layout = StateLayout(**layout_data) if isinstance(layout_data, dict) else None
    use_spatial_cnn = checkpoint.get("network_type") == "spatial_cnn"
    dueling = checkpoint.get("dueling", checkpoint.get("train_cfg", {}).get("dueling", False))
    checkpoint_actions = checkpoint.get("actions")
    if isinstance(checkpoint_actions, dict) and len(checkpoint_actions) != NUM_ACTIONS:
        raise ValueError(
            f"Checkpoint was trained with {len(checkpoint_actions)} actions, but current code expects {NUM_ACTIONS}. "
            "Retrain the DQN with the current action space."
        )
    checkpoint["env_cfg_obj"] = env_cfg
    checkpoint["state_cfg_obj"] = state_cfg

    onnx_path = checkpoint_path.with_suffix('.onnx')
    if not _DISABLE_ONNX_AUTOLOAD and onnx_path.exists():
        print(f"⚡ [ONNX] Auto-loading accelerated model: {onnx_path.name}")
        policy = ONNXPolicyWrapper(str(onnx_path), device)
        return policy, checkpoint

    policy = QNetwork(
        int(checkpoint["state_dim"]),
        hidden_dim=hidden_dim,
        layout=layout,
        use_spatial_cnn=use_spatial_cnn,
        dueling=dueling,
    )
    try:
        policy.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint model shape does not match the current policy architecture. "
            "Retrain the DQN or restore the matching action/state configuration."
        ) from exc
    policy.to(device)
    policy.eval()
    return policy, checkpoint
