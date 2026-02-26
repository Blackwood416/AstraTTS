#!/bin/bash

# AstraTTS 统一发布脚本 (Linux 版)

PublishDir="publish-linux"
Runtime="linux-x64"

if [ -d "$PublishDir" ]; then
    echo "正在清理旧的发布目录..."
    rm -rf "$PublishDir"
fi

echo "🚀 开始发布 AstraTTS.Web (astra-server)..."
dotnet publish AstraTTS.Web/AstraTTS.Web.csproj -c Release -r $Runtime --self-contained true -o "$PublishDir" /p:PublishSingleFile=false /p:AllowMissingPrunePackageData=true

echo "🚀 开始发布 AstraTTS.CLI (astra-cli)..."
dotnet publish AstraTTS.CLI/AstraTTS.CLI.csproj -c Release -r $Runtime --self-contained true -o "$PublishDir" /p:PublishSingleFile=false /p:AllowMissingPrunePackageData=true

# 清理旧的 config.json (如果残留)
rm -f "$PublishDir/config.json"

# 复制必要资源
echo "📦 正在复制资源文件..."
if [ -f "config.template.yaml" ]; then
    cp "config.template.yaml" "$PublishDir/config.template.yaml"
    cp "config.template.yaml" "$PublishDir/config.yaml"
fi

if [ -d "resources-minimal" ]; then
    echo "📦 正在复制资源文件 (minimal)..."
    cp -r resources-minimal "$PublishDir/resources"
fi

if [ -d "tools" ]; then
    echo "📦 正在复制工具目录..."
    cp -r tools "$PublishDir/tools"
    # 删除 Windows 专用的 Python 运行时 (为了在 Linux 环境下精简体积)
    if [ -d "$PublishDir/tools/converter/runtime" ]; then
        echo "🧹 正在清理 Windows Python 运行时..."
        rm -rf "$PublishDir/tools/converter/runtime"
    fi
fi

echo "✅ 发布完成！输出目录: $PublishDir"
chmod +x "$PublishDir/astra-server"
chmod +x "$PublishDir/astra-cli"
