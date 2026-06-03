#!/bin/bash

# ==========================================
# AstraTTS 独立版环境初始化脚本 (macOS)
# 用于：
#   1. 检查/安装运行 TTS 所需的音频播放器 (ffplay 推荐)
#   2. 部署模型转换器必备的 Python 环境
# ==========================================

set -e

echo "🚀 开始初始化 AstraTTS macOS 运行环境..."
echo ""

# 1. 检查 Homebrew (推荐安装方式)
HAS_BREW=0
if command -v brew &> /dev/null; then
    HAS_BREW=1
    echo "✅ 找到 Homebrew: $(brew --version | head -n1)"
else
    echo "⚠️  未检测到 Homebrew (https://brew.sh)。建议安装以便自动安装 ffmpeg。"
fi

# 2. 推荐安装 ffmpeg (提供 ffplay 用于流式播放)
if ! command -v ffplay &> /dev/null; then
    echo ""
    echo "ℹ️  未检测到 ffplay。流式低延迟播放推荐安装 ffmpeg。"
    if [ $HAS_BREW -eq 1 ]; then
        echo "📦 通过 Homebrew 安装 ffmpeg..."
        brew install ffmpeg || echo "⚠️  ffmpeg 安装失败，您仍可使用系统自带的 afplay (非流式)。"
    else
        echo "💡 您可以手动安装: brew install ffmpeg"
        echo "   未安装 ffplay 时，CLI 流式模式将自动降级使用 afplay。"
    fi
else
    echo "✅ 已找到 ffplay: $(which ffplay)"
fi

# 3. 检查 python3 (macOS 自带 python3 可能为 stub，建议 brew install python)
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未安装 python3。请先执行: brew install python"
    exit 1
fi
echo "✅ 找到 python3: $(python3 --version)"

# 4. 创建并激活虚拟环境 (位于 tools/converter/.venv)
VENV_DIR="tools/converter/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 正在创建 Python 虚拟环境 ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
echo "✅ 已激活虚拟环境"

# 5. 安装 Python 依赖 (mac 默认使用官方源，亦可手动改国内源)
echo "📦 正在升级 pip 并安装依赖包 (onnx, numpy, onnxsim, onnxruntime)..."
pip install --upgrade pip
pip install onnx numpy onnxsim onnxruntime

echo "📦 正在安装核心依赖 (CPU 版 PyTorch)..."
pip install torch

echo ""
echo "🎉 初始化完成！模型转换器已准备就绪。"
echo "💡 后续单独激活转换器虚拟环境: source tools/converter/.venv/bin/activate"