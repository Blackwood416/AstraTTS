#!/bin/bash

# AstraTTS 资源与发布整合打包脚本 (Linux)

VERSION="v1.2.1"
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



echo -e "\n🎉 所有打包作业均已进入: $(realpath "$RELEASE_DIR")"
echo -e "提示: 现在可以直接前往 Releases 页面上传 '$RELEASE_DIR' 内的文件了。"
