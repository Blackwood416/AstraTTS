#!/bin/bash

# AstraTTS 资源与发布整合打包脚本 (Linux & macOS)
# 自动检测可用的 publish-* 目录并打包

VERSION="v1.2.1"
RELEASE_DIR="releases"

mkdir -p "$RELEASE_DIR"

echo "📦 开始打包 AstraTTS $VERSION 发布文件..."

# ----------- 1. 打包 Linux 整合包 -----------
if [ -d "publish-linux" ]; then
    LINUX_TAR="$RELEASE_DIR/AstraTTS-$VERSION-linux64.tar.gz"
    rm -f "$LINUX_TAR"
    echo -e "\n正在压缩 Linux 整合包 -> $LINUX_TAR ..."

    cp init-env.sh publish-linux/
    chmod +x publish-linux/init-env.sh

    (cd publish-linux && tar -czf "../$LINUX_TAR" ./*)

    echo "✅ Linux 整合包 (.tar.gz) 打包完成！"
else
    echo -e "\n⚠️  未找到 publish-linux 目录，跳过 Linux 打包。提示: 请先运行 ./publish-linux.sh"
fi

# ----------- 2. 打包 macOS 整合包 (arm64) -----------
pack_mac() {
    local arch="$1"          # arm64 / x64
    local arch_label="$2"    # arm64 / x64 (用于文件名)
    local pub_dir="publish-mac-$arch"
    if [ ! -d "$pub_dir" ]; then
        return
    fi

    local mac_tar="$RELEASE_DIR/AstraTTS-$VERSION-macOS-$arch_label.tar.gz"
    rm -f "$mac_tar"
    echo -e "\n正在压缩 macOS ($arch_label) 整合包 -> $mac_tar ..."

    cp init-env-mac.sh "$pub_dir/"
    chmod +x "$pub_dir/init-env-mac.sh"

    (cd "$pub_dir" && tar -czf "../$mac_tar" ./*)

    echo "✅ macOS ($arch_label) 整合包打包完成！"
}

pack_mac "arm64" "arm64"
pack_mac "x64" "x64"

if [ ! -d "publish-mac-arm64" ] && [ ! -d "publish-mac-x64" ]; then
    echo -e "\n⚠️  未找到 publish-mac-* 目录，跳过 macOS 打包。提示: 请先运行 ./publish-mac.sh"
fi

# ----------- 3. 打包 macOS DMG (若有 publish-mac-* 目录) -----------
if [ -f "pack-mac-dmg.sh" ]; then
    if [ -d "publish-mac-arm64" ] || [ -d "publish-mac-x64" ]; then
        echo -e "\n📦 正在生成 macOS .dmg 安装包..."
        chmod +x pack-mac-dmg.sh
        ./pack-mac-dmg.sh "$VERSION" "$RELEASE_DIR" || echo "⚠️  DMG 打包失败，请检查 pack-mac-dmg.sh 输出。"
    fi
fi

# ----------- 4. 资源包 (resources-minimal) -----------
if [ -d "resources-minimal" ]; then
    RES_TAR="$RELEASE_DIR/AstraTTS-resources-minimal-$VERSION.tar.gz"
    rm -f "$RES_TAR"
    echo -e "\n正在压缩独立资源包 (resources-minimal) -> $RES_TAR ..."
    tar -czf "$RES_TAR" resources-minimal
    echo "✅ 独立资源包打包完成！"
fi

echo -e "\n🎉 所有打包作业均已进入: $(cd "$RELEASE_DIR" && pwd)"
echo -e "提示: 现在可以直接前往 Releases 页面上传 '$RELEASE_DIR' 内的文件了。"