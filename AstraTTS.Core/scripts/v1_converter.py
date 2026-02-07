import os
import sys
import json
import torch
import numpy as np
try:
    import onnx
except ImportError:
    onnx = None

import argparse
import logging
import tempfile
import shutil
from io import BytesIO
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_weights(path, name="Model"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {name} weight file not found: {path}")
    
    filesize = os.path.getsize(path)
    logger.info(f"Loading {name} weights: {os.path.basename(path)} ({filesize / 1024 / 1024:.2f} MB)...")
    
    if filesize == 0:
        raise ValueError(f"❌ {name} weight file is EMPTY (0 bytes): {path}. Please check your download.")

    try:
        # Check for missing 'PK' header (common in some SoVITS formats)
        with open(path, "rb") as f:
            meta = f.read(2)
            if meta != b"PK":
                logger.warning(f"  Missing 'PK' header in {name}. Attempting to patch in-memory...")
                data = b"PK" + f.read()
                bio = BytesIO(data)
                checkpoint = torch.load(bio, map_location='cpu', weights_only=False)
            else:
                # Try weights_only=True first (Safer for newer models)
                try:
                    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                except Exception:
                    logger.warning(f"  Standard safe load failed for {name}, trying legacy load...")
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e2:
        if "stack underflow" in str(e2).lower() or "pickle" in str(e2).lower():
            logger.error(f"❌ CRITICAL ERROR: Failed to unpickle {name} weights.")
            logger.error(f"   This usually means the file is CORRUPTED or missing a valid header.")
            logger.error(f"   Note: We attempted to auto-fix missing 'PK' headers, but it still failed.")
        raise e2

    # Handle different SoVITS save formats
    if isinstance(checkpoint, dict):
        if 'weight' in checkpoint:
            return checkpoint['weight']
        # Maybe it's a Lightning checkpoint (GPT)
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        return checkpoint # Assume it's the raw state_dict
    return checkpoint

# ==============================================================================
# UTILS MOCKING (Required for torch.load to unpickle SoVITS models)
# ==============================================================================

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self): return self.__dict__.keys()
    def items(self): return self.__dict__.items()
    def values(self): return self.__dict__.values()
    def __len__(self): return len(self.__dict__)
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): return setattr(self, key, value)
    def __contains__(self, key): return key in self.__dict__
    def __repr__(self): return self.__dict__.__repr__()

import types
utils_mock = types.ModuleType("utils")
utils_mock.HParams = HParams
sys.modules["utils"] = utils_mock

# ==============================================================================
# KEY GENERATION HELPERS
# ==============================================================================

def get_t2s_keys():
    keys = ["ar_audio_embedding.word_embeddings.weight", "ar_audio_position.alpha"]
    for i in range(24):
        keys.extend([
            f"transformer_encoder.layers.{i}.self_attn.in_proj_weight",
            f"transformer_encoder.layers.{i}.self_attn.in_proj_bias",
            f"transformer_encoder.layers.{i}.self_attn.out_proj.weight",
            f"transformer_encoder.layers.{i}.self_attn.out_proj.bias",
            f"transformer_encoder.layers.{i}.linear1.weight",
            f"transformer_encoder.layers.{i}.linear1.bias",
            f"transformer_encoder.layers.{i}.linear2.weight",
            f"transformer_encoder.layers.{i}.linear2.bias",
            f"transformer_encoder.layers.{i}.norm1.weight",
            f"transformer_encoder.layers.{i}.norm1.bias",
            f"transformer_encoder.layers.{i}.norm2.weight",
            f"transformer_encoder.layers.{i}.norm2.bias",
        ])
    keys.append("ar_predict_layer.weight")
    return keys

def get_vits_keys():
    keys = ["vq_model.enc_p.ssl_proj.weight", "vq_model.enc_p.ssl_proj.bias"]
    for i in range(3):
        keys.extend([
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.emb_rel_k",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.emb_rel_v",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.conv_q.weight",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.conv_q.bias",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.conv_k.weight",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.conv_k.bias",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.conv_v.weight",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.conv_v.bias",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.conv_o.weight",
            f"vq_model.enc_p.encoder_ssl.attn_layers.{i}.conv_o.bias"
        ])
    for i in range(3):
        keys.extend([
            f"vq_model.enc_p.encoder_ssl.norm_layers_1.{i}.gamma",
            f"vq_model.enc_p.encoder_ssl.norm_layers_1.{i}.beta",
            f"vq_model.enc_p.encoder_ssl.ffn_layers.{i}.conv_1.weight",
            f"vq_model.enc_p.encoder_ssl.ffn_layers.{i}.conv_1.bias",
            f"vq_model.enc_p.encoder_ssl.ffn_layers.{i}.conv_2.weight",
            f"vq_model.enc_p.encoder_ssl.ffn_layers.{i}.conv_2.bias",
            f"vq_model.enc_p.encoder_ssl.norm_layers_2.{i}.gamma",
            f"vq_model.enc_p.encoder_ssl.norm_layers_2.{i}.beta"
        ])
    for i in range(6):
        keys.extend([
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.emb_rel_k",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.emb_rel_v",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.conv_q.weight",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.conv_q.bias",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.conv_k.weight",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.conv_k.bias",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.conv_v.weight",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.conv_v.bias",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.conv_o.weight",
            f"vq_model.enc_p.encoder_text.attn_layers.{i}.conv_o.bias"
        ])
    for i in range(6):
        keys.extend([
            f"vq_model.enc_p.encoder_text.norm_layers_1.{i}.gamma",
            f"vq_model.enc_p.encoder_text.norm_layers_1.{i}.beta",
            f"vq_model.enc_p.encoder_text.ffn_layers.{i}.conv_1.weight",
            f"vq_model.enc_p.encoder_text.ffn_layers.{i}.conv_1.bias",
            f"vq_model.enc_p.encoder_text.ffn_layers.{i}.conv_2.weight",
            f"vq_model.enc_p.encoder_text.ffn_layers.{i}.conv_2.bias",
            f"vq_model.enc_p.encoder_text.norm_layers_2.{i}.gamma",
            f"vq_model.enc_p.encoder_text.norm_layers_2.{i}.beta"
        ])
    keys.extend(["vq_model.enc_p.text_embedding.weight", 
                 "vq_model.enc_p.mrte.cross_attention.conv_q.weight", "vq_model.enc_p.mrte.cross_attention.conv_q.bias",
                 "vq_model.enc_p.mrte.cross_attention.conv_k.weight", "vq_model.enc_p.mrte.cross_attention.conv_k.bias",
                 "vq_model.enc_p.mrte.cross_attention.conv_v.weight", "vq_model.enc_p.mrte.cross_attention.conv_v.bias",
                 "vq_model.enc_p.mrte.cross_attention.conv_o.weight", "vq_model.enc_p.mrte.cross_attention.conv_o.bias",
                 "vq_model.enc_p.mrte.c_pre.weight", "vq_model.enc_p.mrte.c_pre.bias",
                 "vq_model.enc_p.mrte.text_pre.weight", "vq_model.enc_p.mrte.text_pre.bias",
                 "vq_model.enc_p.mrte.c_post.weight", "vq_model.enc_p.mrte.c_post.bias"])
    for i in range(3):
        keys.extend([
            f"vq_model.enc_p.encoder2.attn_layers.{i}.emb_rel_k",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.emb_rel_v",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.conv_q.weight",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.conv_q.bias",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.conv_k.weight",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.conv_k.bias",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.conv_v.weight",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.conv_v.bias",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.conv_o.weight",
            f"vq_model.enc_p.encoder2.attn_layers.{i}.conv_o.bias"
        ])
    for i in range(3):
        keys.extend([
            f"vq_model.enc_p.encoder2.norm_layers_1.{i}.gamma",
            f"vq_model.enc_p.encoder2.norm_layers_1.{i}.beta",
            f"vq_model.enc_p.encoder2.ffn_layers.{i}.conv_1.weight",
            f"vq_model.enc_p.encoder2.ffn_layers.{i}.conv_1.bias",
            f"vq_model.enc_p.encoder2.ffn_layers.{i}.conv_2.weight",
            f"vq_model.enc_p.encoder2.ffn_layers.{i}.conv_2.bias",
            f"vq_model.enc_p.encoder2.norm_layers_2.{i}.gamma",
            f"vq_model.enc_p.encoder2.norm_layers_2.{i}.beta"
        ])
    keys.extend(["vq_model.enc_p.proj.weight", "vq_model.enc_p.proj.bias",
                 "vq_model.dec.conv_pre.weight", "vq_model.dec.conv_pre.bias"])
    for i in range(5):
        keys.extend([f"vq_model.dec.ups.{i}.bias", f"vq_model.dec.ups.{i}.weight_g", f"vq_model.dec.ups.{i}.weight_v"])
    for i in range(15):
        for c in [1, 2]:
            for j in range(3):
                keys.extend([f"vq_model.dec.resblocks.{i}.convs{c}.{j}.bias",
                             f"vq_model.dec.resblocks.{i}.convs{c}.{j}.weight_g",
                             f"vq_model.dec.resblocks.{i}.convs{c}.{j}.weight_v"])
    keys.extend(["vq_model.dec.conv_post.weight", "vq_model.dec.cond.weight", "vq_model.dec.cond.bias"])
    for f in [0, 2, 4, 6]:
        keys.extend([f"vq_model.flow.flows.{f}.pre.weight", f"vq_model.flow.flows.{f}.pre.bias"])
        for i in range(4):
            keys.extend([f"vq_model.flow.flows.{f}.enc.in_layers.{i}.bias",
                         f"vq_model.flow.flows.{f}.enc.in_layers.{i}.weight_g",
                         f"vq_model.flow.flows.{f}.enc.in_layers.{i}.weight_v",
                         f"vq_model.flow.flows.{f}.enc.res_skip_layers.{i}.bias",
                         f"vq_model.flow.flows.{f}.enc.res_skip_layers.{i}.weight_g",
                         f"vq_model.flow.flows.{f}.enc.res_skip_layers.{i}.weight_v"])
        keys.extend([f"vq_model.flow.flows.{f}.enc.cond_layer.bias",
                     f"vq_model.flow.flows.{f}.enc.cond_layer.weight_g",
                     f"vq_model.flow.flows.{f}.enc.cond_layer.weight_v",
                     f"vq_model.flow.flows.{f}.post.weight",
                     f"vq_model.flow.flows.{f}.post.bias"])
    keys.append("vq_model.quantizer.vq.layers.0._codebook.embed")
    return keys

def get_prompt_encoder_keys():
    return [
        "ref_enc.spectral.0.fc.weight", "ref_enc.spectral.0.fc.bias",
        "ref_enc.spectral.3.fc.weight", "ref_enc.spectral.3.fc.bias",
        "ref_enc.temporal.0.conv1.conv.weight", "ref_enc.temporal.0.conv1.conv.bias",
        "ref_enc.temporal.1.conv1.conv.weight", "ref_enc.temporal.1.conv1.conv.bias",
        "ref_enc.slf_attn.w_qs.weight", "ref_enc.slf_attn.w_qs.bias",
        "ref_enc.slf_attn.w_ks.weight", "ref_enc.slf_attn.w_ks.bias",
        "ref_enc.slf_attn.w_vs.weight", "ref_enc.slf_attn.w_vs.bias",
        "ref_enc.slf_attn.fc.weight", "ref_enc.slf_attn.fc.bias",
        "ref_enc.fc.fc.weight", "ref_enc.fc.fc.bias",
        "sv_emb.weight", "sv_emb.bias",
        "ge_to512.weight", "ge_to512.bias",
        "prelu.weight"
    ]

ENCODER_KEYS = [
    "encoder.ar_text_embedding.word_embeddings.weight",
    "encoder.bert_proj.weight",
    "encoder.bert_proj.bias",
    "encoder.ar_text_position.alpha",
    "vits.ssl_proj.weight",
    "vits.ssl_proj.bias",
    "vits.quantizer.vq.layers.0._codebook.embed"
]

# ==============================================================================
# CORE PIPELINE FUNCTIONS
# ==============================================================================

def patch_and_save_embedded(shell_path, weight_map, output_path, is_fp16=True):
    logger.info(f"Embedding weights into {os.path.basename(shell_path)} -> {os.path.basename(output_path)}")
    model = onnx.load_model(shell_path, load_external_data=False)
    initializer_map = {init.name: init for init in model.graph.initializer}
    
    count = 0
    for key, tensor in weight_map.items():
        if key in initializer_map:
            tensor_proto = initializer_map[key]
            if is_fp16:
                numpy_array = tensor.to(torch.float16).cpu().numpy().astype(np.float32)
            else:
                numpy_array = tensor.to(torch.float32).cpu().numpy()
                
            proto_size = 1
            for d in list(tensor_proto.dims): proto_size *= d
            numpy_size = numpy_array.size
            
            if proto_size != numpy_size:
                logger.error(f"❌ SHAPE MISMATCH for {key}: Shell expects size {proto_size}, got {numpy_size}")
            
            tensor_proto.raw_data = numpy_array.tobytes()
            del tensor_proto.external_data[:]
            tensor_proto.data_location = onnx.TensorProto.DEFAULT
            count += 1
    
    unpatched_count = sum(1 for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL)
    if unpatched_count > 0:
        logger.error(f"❌ {unpatched_count} tensors were NOT patched!")
            
    onnx.save(model, output_path)
    logger.info(f"  Embedded {count} tensors.")

def simplify_model(input_path: str, output_path: str):
    try:
        from onnxsim import simplify
        logger.info(f"Simplifying: {os.path.basename(input_path)}")
        model = onnx.load(input_path)
        model_simp, check = simplify(model)
        onnx.save(model_simp, output_path)
        return True
    except ImportError:
        logger.warning("onnxsim not found, skipping simplification")
        return False
    except Exception as e:
        logger.error(f"Simplification failed: {e}")
        return False

def quantize_model_dynamic(input_path: str, output_path: str):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        logger.info(f"Quantizing: {os.path.basename(input_path)}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            preprocessed_path = os.path.join(tmp_dir, "preprocessed.onnx")
            try:
                quant_pre_process(input_model_path=input_path, output_model_path=preprocessed_path, skip_symbolic_shape=True)
            except:
                shutil.copy(input_path, preprocessed_path)
                
            quantize_dynamic(
                model_input=preprocessed_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                per_channel=False,
                reduce_range=False,
                extra_options={"ActivationSymmetric": False, "WeightSymmetric": True, "ForceQuantizeNoInputCheck": True},
                op_types_to_quantize=["MatMul", "Gemm", "Attention", "LSTM", "GRU"]
            )
        return True
    except ImportError:
        logger.warning("onnxruntime and its quantization tools not found, skipping quantization")
        return False
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return False

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def run_conversion(args):
    os.makedirs(args.out, exist_ok=True)
    
    logger.info("Loading PyTorch weights...")
    gpt_weights = load_weights(args.ckpt, "GPT")
    sovits_weights = load_weights(args.pth, "SoVITS")
    
    work_items = [
        {
            "name": "t2s_encoder",
            "shell": "t2s_encoder_fp32.onnx",
            "is_fp16": False,
            "map": lambda: {
                **{f"encoder.{k[6:]}": gpt_weights[k] for k in gpt_weights if k.startswith("model.ar_") or k.startswith("model.bert_")},
                **{"vits.ssl_proj.weight": sovits_weights["ssl_proj.weight"], 
                   "vits.ssl_proj.bias": sovits_weights["ssl_proj.bias"],
                   "vits.quantizer.vq.layers.0._codebook.embed": sovits_weights["quantizer.vq.layers.0._codebook.embed"]}
            }
        },
        {
            "name": "t2s_first_stage_decoder",
            "shell": "t2s_first_stage_decoder_fp32.onnx",
            "is_fp16": True,
            "map": lambda: {k: gpt_weights[f"model.{k.replace('transformer_encoder', 'h')}"] 
                           for k in get_t2s_keys() if f"model.{k.replace('transformer_encoder', 'h')}" in gpt_weights}
        },
        {
            "name": "t2s_stage_decoder",
            "shell": "t2s_stage_decoder_fp32.onnx",
            "is_fp16": True,
            "map": lambda: {k: gpt_weights[f"model.{k.replace('transformer_encoder', 'h')}"] 
                           for k in get_t2s_keys() if f"model.{k.replace('transformer_encoder', 'h')}" in gpt_weights}
        },
        {
            "name": "vits",
            "shell": "vits_fp32.onnx",
            "is_fp16": True,
            "map": lambda: {k: sovits_weights[k[9:] if k.startswith("vq_model.") else k] 
                           for k in get_vits_keys() if (k[9:] if k.startswith("vq_model.") else k) in sovits_weights}
        },
        {
            "name": "prompt_encoder",
            "shell": "prompt_encoder_fp32.onnx",
            "is_fp16": True,
            "map": lambda: {k: sovits_weights[k] for k in get_prompt_encoder_keys() if k in sovits_weights}
        }
    ]

    for item in work_items:
        shell_path = os.path.join(args.shells, item["shell"])
        if not os.path.exists(shell_path):
            logger.warning(f"Shell not found: {shell_path}, skipping {item['name']}")
            continue
            
        name = item["name"]
        # 定义路径
        fp32_path = os.path.join(args.out, f"{name}_fp32.onnx")
        int8_path = os.path.join(args.out, f"{name}_int8.onnx")
        opt_path = os.path.join(args.out, f"{name}_opt.onnx")
        final_path = os.path.join(args.out, f"{name}.onnx")
        
        # 1. 导出基础 FP32 模型
        patch_and_save_embedded(shell_path, item["map"](), fp32_path, is_fp16=item["is_fp16"])
        current_path = fp32_path
        
        # 2. 可选：动态量化 (INT8) - 用户要求优先量化
        if args.quantize:
            if quantize_model_dynamic(current_path, int8_path):
                current_path = int8_path
            else:
                logger.warning(f"Quantization failed for {name}, falling back to previous stage.")
                
        # 3. 可选：模型简化 (Simplify) - 在量化后的基础上简化效果更佳
        if args.simplify:
            if simplify_model(current_path, opt_path):
                current_path = opt_path
            else:
                logger.warning(f"Simplification failed for {name}, falling back to previous stage.")
        
        # 4. 最终定稿：将最新结果命名为标准文件名
        logger.info(f"Finalizing {name}: {os.path.basename(current_path)} -> {os.path.basename(final_path)}")
        if os.path.exists(final_path): os.remove(final_path)
        shutil.copy(current_path, final_path)
                
        # 5. 清理中间文件
        if args.clean:
            logger.info(f"Cleaning intermediate files for {name}...")
            for p in [fp32_path, int8_path, opt_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as e:
                        logger.warning(f"Failed to remove {p}: {e}")

    logger.info("🎉 All models processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AstraTTS V1 Unified Converter")
    parser.add_argument("--ckpt", required=True, help="Path to GPT .ckpt file")
    parser.add_argument("--pth", required=True, help="Path to SoVITS .pth file")
    parser.add_argument("--shells", required=True, help="Directory containing template ONNX shells")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--simplify", action="store_true", help="Enable model simplification")
    parser.add_argument("--quantize", action="store_true", help="Enable dynamic INT8 quantization")
    parser.add_argument("--clean", action="store_true", help="Remove intermediate files")

    args = parser.parse_args()
    run_conversion(args)
