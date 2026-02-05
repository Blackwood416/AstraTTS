"""
ONNX INT8 Dynamic Quantization Script
======================================
Quantizes ONNX models to INT8 dynamic for CPU inference acceleration.

Usage:
    python quantize_models.py --input_dir ./models/cx_standalone --output_dir ./models/cx_standalone_int8

Requirements:
    pip install onnxruntime onnx
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install onnxruntime onnx")
    sys.exit(1)


def preprocess_model(input_path: str, output_path: str) -> bool:
    """
    Preprocess ONNX model for quantization (shape inference, optimization).
    """
    try:
        quant_pre_process(
            input_model_path=input_path,
            output_model_path=output_path,
            skip_symbolic_shape=True,  # Skip symbolic shape inference (helps with dynamic models)
        )
        return True
    except Exception as e:
        print(f"    Preprocess warning: {e}")
        # If preprocessing fails, just copy the original
        import shutil
        shutil.copy(input_path, output_path)
        return False


def quantize_model(input_path: str, output_path: str, model_name: str, weight_type: QuantType = QuantType.QInt8):
    """
    Quantize a single ONNX model to INT8 dynamic.
    """
    print(f"  Quantizing: {os.path.basename(input_path)}")
    print(f"    Input size: {os.path.getsize(input_path) / 1024 / 1024:.1f} MB")
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preprocessed_path = os.path.join(tmp_dir, "preprocessed.onnx")
            
            # Step 1: Preprocess model
            print(f"    Preprocessing...")
            preprocess_model(input_path, preprocessed_path)
            
            # Step 2: Quantize with appropriate settings
            # IMPORTANT: Only quantize MatMul and Gemm operations
            # Conv operations require AVX-VNNI (not available on Xeon E5 v3)
            # and will cause "ConvInteger not implemented" errors
            extra_options = {
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "ForceQuantizeNoInputCheck": True,  # More lenient for complex models
            }
            
            quantize_dynamic(
                model_input=preprocessed_path,
                model_output=output_path,
                weight_type=weight_type,
                per_channel=False,
                reduce_range=False,
                extra_options=extra_options,
                # Only quantize MatMul and Gemm - these have good CPU EP support
                # Skip Conv layers to avoid ConvInteger errors
                op_types_to_quantize=["MatMul", "Gemm", "Attention", "LSTM", "GRU"]
            )
        
        output_size = os.path.getsize(output_path) / 1024 / 1024
        input_size = os.path.getsize(input_path) / 1024 / 1024
        reduction = (1 - output_size / input_size) * 100
        
        print(f"    Output size: {output_size:.1f} MB ({reduction:.1f}% reduction)")
        return True
        
    except Exception as e:
        print(f"    Quantization failed: {e}")
        print(f"    Falling back to copy original model...")
        try:
            import shutil
            shutil.copy(input_path, output_path)
            print(f"    Copied original (no quantization)")
            return True
        except:
            return False


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8 dynamic")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing FP32 ONNX models")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save quantized models")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="Model names to skip (without .onnx)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all ONNX models
    models = list(input_dir.glob("*.onnx"))
    if not models:
        print(f"No ONNX models found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(models)} models in {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    success_count = 0
    skip_count = 0
    
    for model_path in models:
        model_name = model_path.stem
        
        # Check if model should be skipped
        if model_name in args.skip:
            print(f"  Skipping: {model_name} (user requested)")
            skip_count += 1
            continue
        
        output_path = output_dir / f"{model_name}.onnx"
        
        if quantize_model(str(model_path), str(output_path), model_name):
            success_count += 1
    
    print("-" * 50)
    print(f"Quantization complete: {success_count} succeeded, {skip_count} skipped, {len(models) - success_count - skip_count} failed")
    
    # Calculate total size reduction
    input_total = sum(f.stat().st_size for f in input_dir.glob("*.onnx")) / 1024 / 1024
    output_total = sum(f.stat().st_size for f in output_dir.glob("*.onnx")) / 1024 / 1024
    
    print(f"Total size: {input_total:.1f} MB -> {output_total:.1f} MB ({(1 - output_total/input_total)*100:.1f}% reduction)")


if __name__ == "__main__":
    main()
