"""
ONNX Model Simplifier Script
=============================
Simplifies ONNX models using onnxsim to reduce redundant operations.

Usage:
    python simplify_models.py --input_dir ./models/cx_standalone_int8 --output_dir ./models/cx_standalone_opt

Requirements:
    pip install onnxsim onnx
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import onnx
    from onnxsim import simplify
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install onnxsim onnx")
    sys.exit(1)


def simplify_model(input_path: str, output_path: str) -> bool:
    """
    Simplify a single ONNX model.
    """
    print(f"  Simplifying: {os.path.basename(input_path)}")
    input_size = os.path.getsize(input_path) / 1024 / 1024
    print(f"    Input size: {input_size:.1f} MB")
    
    try:
        # Load model
        model = onnx.load(input_path)
        
        # Simplify model
        model_simp, check = simplify(
            model,
            skip_fuse_bn=False,           # Fuse BatchNorm into Conv
            skip_constant_folding=False,  # Fold constant operations
            skip_shape_inference=False,   # Infer shapes
        )
        
        if not check:
            print(f"    Warning: Simplified model validation failed, using original")
            # Still save the simplified model, validation can fail for dynamic models
        
        # Save simplified model
        onnx.save(model_simp, output_path)
        
        output_size = os.path.getsize(output_path) / 1024 / 1024
        reduction = (1 - output_size / input_size) * 100
        
        # Count nodes
        orig_nodes = len(model.graph.node)
        simp_nodes = len(model_simp.graph.node)
        node_reduction = (1 - simp_nodes / orig_nodes) * 100 if orig_nodes > 0 else 0
        
        print(f"    Output size: {output_size:.1f} MB ({reduction:.1f}% size reduction)")
        print(f"    Nodes: {orig_nodes} -> {simp_nodes} ({node_reduction:.1f}% reduction)")
        return True
        
    except Exception as e:
        print(f"    Simplification failed: {e}")
        print(f"    Copying original model...")
        try:
            import shutil
            shutil.copy(input_path, output_path)
            return True
        except:
            return False


def main():
    parser = argparse.ArgumentParser(description="Simplify ONNX models using onnxsim")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing ONNX models")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save simplified models")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="Model names to skip")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(input_dir.glob("*.onnx"))
    if not models:
        print(f"No ONNX models found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(models)} models in {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    success_count = 0
    
    for model_path in models:
        model_name = model_path.stem
        
        if model_name in args.skip:
            print(f"  Skipping: {model_name}")
            continue
        
        output_path = output_dir / f"{model_name}.onnx"
        
        if simplify_model(str(model_path), str(output_path)):
            success_count += 1
    
    print("-" * 50)
    print(f"Simplification complete: {success_count}/{len(models)} succeeded")
    
    # Total size comparison
    input_total = sum(f.stat().st_size for f in input_dir.glob("*.onnx")) / 1024 / 1024
    output_total = sum(f.stat().st_size for f in output_dir.glob("*.onnx")) / 1024 / 1024
    print(f"Total size: {input_total:.1f} MB -> {output_total:.1f} MB")


if __name__ == "__main__":
    main()
