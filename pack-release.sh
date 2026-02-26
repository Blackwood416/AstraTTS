#!/bin/bash

# AstraTTS 资源与发布整合打包脚本 (Linux)

VERSION="v1.1.1"
RELEASE_DIR="releases"

# 创建目录
mkdir -p "$RELEASE_DIR"

echo "📦 开始打包 AstraTTS $VERSION 发布文件..."

# 1. 打包 Linux 整合包
if [ -d "publish-linux" ]; then
    LINUX_TAR="$RELEASE_DIR/AstraTTS-$VERSION-linux64.tar.gz"
    rm -f "$LINUX_TAR"
    echo -e "\n正在压缩 Linux 整合包 -> $LINUX_TAR ..."
    
    # 拷贝初始化脚本到发布目录
    cp init-env.sh publish-linux/
    chmod +x publish-linux/init-env.sh
    
    # 切换到上级目录，将 publish-linux 作为根目录打包
    cd publish-linux || exit
    tar -czvf "../$LINUX_TAR" ./* > /dev/null
    cd ..
    
    echo "✅ Linux 整合包 (.tar.gz) 打包完成！"
else
    echo -e "\n⚠️ 未找到 publish-linux 目录，跳过打包 Linux。提示: 请先运行 ./publish-linux.sh"
fi

# 2. 单独打包核心资源依赖库
if [ -d "resources-minimal" ]; then
    RES_ZIP="$RELEASE_DIR/AstraTTS-resources-minimal-$VERSION.zip"
    rm -f "$RES_ZIP"
    echo -e "\n正在压缩独立资源包 (resources-minimal) -> $RES_ZIP ..."
    
    zip -r "$RES_ZIP" resources-minimal > /dev/null
    
    echo "✅ 独立资源包打包完成！"
else
    echo -e "\n⚠️ 未找到 resources-minimal 目录，跳过核心资源的单独打包。"
fi

echo -e "\n🎉 所有打包作业均已进入: $(realpath "$RELEASE_DIR")"
echo -e "提示: 现在可以直接前往 Releases 页面上传 '$RELEASE_DIR' 内的文件了。"
