#!/bin/bash

# ==========================================
# AstraTTS 独立版环境初始化脚本 (Linux)
# 用于模型转换器必备的 Python 环境部署
# ==========================================

echo "🚀 开始初始化 AstraTTS 模型转换器 Python 环境..."

# 1. 检查 python3 及其模块
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未安装 python3。请先执行 'sudo apt-get install python3 python3-pip python3-venv' (Ubuntu/Debian) 或相应的包管理命令。"
    exit 1
fi

echo "✅ 找到 python3: $(python3 --version)"

# 2. 创建并激活虚拟环境 (位于 tools/converter/.venv)
VENV_DIR="tools/converter/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 正在创建 Python 虚拟环境 ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 无法创建虚拟环境。你可能需要安装 'python3-venv' 包。"
        exit 1
    fi
else
    echo "✅ 虚拟环境已存在: $VENV_DIR"
fi

# 3. 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo "✅ 已激活虚拟环境"

# 4. 更新 pip 并安装依赖 (使用国内清华源/阿里源加速)
echo "📦 正在安装依赖包 (onnx, numpy, onnxsim, onnxruntime)..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx numpy onnxsim onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "📦 正在安装核心依赖 (CPU 版 PyTorch) 以减小体积..."
pip install torch -f https://mirrors.aliyun.com/pytorch-wheels/cpu

echo ""
echo "🎉 初始化完成！模型转换器已经准备就绪。"
echo "💡 如果需要在后台单独运行转换器，请先执行: source tools/converter/.venv/bin/activate"
