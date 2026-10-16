"""
Export PyTorch GeneratorFunieGAN to ONNX (dynamic batch dim).

Usage:
  python tools/onnx_export.py --model-path PyTorch/models/funie_generator.pth --out model/funie.onnx

Dependencies:
  pip install torch 

Notes:
  - This script attempts common saved-state patterns ('state_dict', 'model', 'netG').
  - Exports with input name "input" and output name "output" and dynamic batch axis.
"""
import argparse
import os
import sys

# ensure PyTorch module is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))  # tools/
_repo_root = os.path.dirname(_script_dir)  # FUnIE-GAN/
sys.path.insert(0, _repo_root)

import torch

try:
    from PyTorch.nets.funiegan import GeneratorFunieGAN
except Exception as e:
    print("ERROR: Failed to import GeneratorFunieGAN from PyTorch.nets.funiegan:", e)
    sys.exit(1)


def load_state_dict_flexible(path, model):
    """
    Load weights supporting different .pth layout conventions:
      - plain state_dict
      - dict with key 'state_dict'
      - dict with key 'model' or 'netG' or 'generator'
    """
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        # common keys
        for key in ("state_dict", "model", "netG", "generator"):
            if key in state:
                sd = state[key]
                # sometimes keys are prefixed with module.
                if all(k.startswith("module.") for k in sd.keys()):
                    sd = {k.replace("module.", ""): v for k, v in sd.items()}
                model.load_state_dict(sd)
                return
        # If dict looks like a state_dict (has tensor values), try to load directly
        if all(isinstance(v, torch.Tensor) for v in state.values()):
            sd = state
            if all(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd)
            return
        # else fallthrough to error
        raise RuntimeError("Unrecognized checkpoint dict format. Expected keys: 'state_dict'/'model'/'netG' or a plain state_dict.")
    else:
        # Could be a SavedModule or other - try direct load_state_dict if possible
        try:
            model.load_state_dict(state)
            return
        except Exception as e:
            raise RuntimeError(f"Unable to load checkpoint from {path}: {e}")


def export_onnx(model_path, out_path, opset=12):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    # instantiate model
    model = GeneratorFunieGAN()
    # load weights
    try:
        load_state_dict_flexible(model_path, model)
    except Exception as e:
        raise RuntimeError(f"Error loading weights: {e}")
    model.eval()
    # dummy input: 1 x 3 x 256 x 256 on CPU
    dummy = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")
    print(f"ONNX export successful: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Export GeneratorFunieGAN -> ONNX")
    p.add_argument("--model-path", type=str, default="PyTorch/models/funie_generator.pth", help="Path to .pth checkpoint")
    p.add_argument("--out", type=str, default="model/funie.onnx", help="Output ONNX path")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version to use")
    args = p.parse_args()
    try:
        export_onnx(args.model_path, args.out, opset=args.opset)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
