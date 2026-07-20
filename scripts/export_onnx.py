import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rl_sahi.rl.checkpoint import load_policy

def main():
    parser = argparse.ArgumentParser(description="Export RL Policy to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found.")
        sys.exit(1)

    print(f"Loading policy from {checkpoint_path}...")
    device = torch.device('cpu')
    
    import rl_sahi.rl.checkpoint
    # Temporarily disable ONNX auto-loading so we load the PyTorch model to export it
    rl_sahi.rl.checkpoint._DISABLE_ONNX_AUTOLOAD = True
    
    policy, checkpoint_data = load_policy(checkpoint_path, device=device)

    input_dim = checkpoint_data["state_dim"]
    dummy_input = torch.randn(1, input_dim, device=device)
    
    onnx_path = checkpoint_path.with_suffix('.onnx')
    print(f"Exporting to {onnx_path}...")
    
    torch.onnx.export(
        policy, 
        dummy_input, 
        str(onnx_path), 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Export successful!")
    print(f"You can now run inference as usual. The system will automatically use the faster ONNX model.")

if __name__ == "__main__":
    main()
