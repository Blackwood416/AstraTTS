#!/bin/bash

# ==========================================
# AstraTTS 统一发布脚本 (macOS 版)
# 用法:
#   ./publish-mac.sh                # 自动检测主机架构 (Apple Silicon -> arm64, Intel -> x64)
#   ./publish-mac.sh arm64          # 强制构建 osx-arm64
#   ./publish-mac.sh x64            # 强制构建 osx-x64
# ==========================================

set -e

# 解析架构参数
ARCH_ARG="${1:-auto}"
if [ "$ARCH_ARG" = "auto" ]; then
    HOST_ARCH=$(uname -m)
    case "$HOST_ARCH" in
        arm64|aarch64) RID="osx-arm64" ;;
        x86_64)        RID="osx-x64" ;;
        *)             RID="osx-arm64" ;;
    esac
elif [ "$ARCH_ARG" = "arm64" ] || [ "$ARCH_ARG" = "aarch64" ]; then
    RID="osx-arm64"
elif [ "$ARCH_ARG" = "x64" ] || [ "$ARCH_ARG" = "x86_64" ]; then
    RID="osx-x64"
else
    echo "❌ 未识别的架构参数: $ARCH_ARG (合法值: auto | arm64 | x64)"
    exit 1
fi

PublishDir="publish-mac-${RID#osx-}"

echo "🚀 目标平台: $RID"
echo "📂 输出目录: $PublishDir"

if [ -d "$PublishDir" ]; then
    echo "🧹 正在清理旧的发布目录..."
    rm -rf "$PublishDir"
fi

# 检查 dotnet
if ! command -v dotnet &> /dev/null; then
    echo "❌ 未检测到 dotnet。请先安装 .NET 10 SDK: https://dotnet.microsoft.com/download"
    exit 1
fi

echo ""
echo "🚀 开始发布 AstraTTS.Web (astra-server)..."
dotnet publish AstraTTS.Web/AstraTTS.Web.csproj \
    -c Release -r "$RID" --self-contained true \
    -o "$PublishDir" \
    /p:PublishSingleFile=false \
    /p:AllowMissingPrunePackageData=true

echo ""
echo "🚀 开始发布 AstraTTS.CLI (astra-cli)..."
dotnet publish AstraTTS.CLI/AstraTTS.CLI.csproj \
    -c Release -r "$RID" --self-contained true \
    -o "$PublishDir" \
    /p:PublishSingleFile=false \
    /p:AllowMissingPrunePackageData=true

# 清理旧的 config.json (如果残留)
rm -f "$PublishDir/config.json"

# 复制必要资源
echo ""
echo "📦 正在复制配置文件..."
if [ -f "config.template.yaml" ]; then
    cp "config.template.yaml" "$PublishDir/config.template.yaml"
    cp "config.template.yaml" "$PublishDir/config.yaml"
fi

if [ -d "resources-minimal" ]; then
    echo "📦 正在复制资源文件 (minimal)..."
    cp -r resources-minimal "$PublishDir/resources"
fi

if [ -d "tools" ]; then
    echo "📦 正在复制工具目录 (converter)..."
    mkdir -p "$PublishDir/tools/converter"
    cp tools/converter/v1_converter.py "$PublishDir/tools/converter/v1_converter.py"
    if [ -d "tools/converter/templates" ]; then
        cp -r tools/converter/templates "$PublishDir/tools/converter/templates"
    fi
    # macOS 不需要 Windows 专用 Python runtime
    if [ -d "$PublishDir/tools/converter/runtime" ]; then
        rm -rf "$PublishDir/tools/converter/runtime"
    fi
fi

# 拷贝 mac 初始化脚本
if [ -f "init-env-mac.sh" ]; then
    cp init-env-mac.sh "$PublishDir/init-env-mac.sh"
    chmod +x "$PublishDir/init-env-mac.sh"
fi

# 设置可执行权限
chmod +x "$PublishDir/astra-server" 2>/dev/null || true
chmod +x "$PublishDir/astra-cli" 2>/dev/null || true

# macOS 上的 Gatekeeper 隔离属性可能阻止运行，剥离 quarantine 属性以方便本机直接使用
if command -v xattr &> /dev/null; then
    xattr -dr com.apple.quarantine "$PublishDir" 2>/dev/null || true
fi

echo ""
echo "✅ 发布完成！输出目录: $(cd "$PublishDir" && pwd)"
echo "运行提示:"
echo "  - 启动 Web 服务: cd $PublishDir && ./astra-server"
echo "  - 启动 CLI 工具: cd $PublishDir && ./astra-cli"
echo ""
echo "💡 若首次运行被 Gatekeeper 阻止，请执行:"
echo "   xattr -dr com.apple.quarantine $PublishDir"