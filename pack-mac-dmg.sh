#!/bin/bash

# ==========================================
# AstraTTS macOS DMG 打包脚本
# 将 publish-mac-* 目录组装为 .app Bundle 并打包为 .dmg 安装包
#
# 用法:
#   ./pack-mac-dmg.sh [VERSION] [OUTPUT_DIR]
#   默认 VERSION=v1.2.1, OUTPUT_DIR=releases
# ==========================================

set -e

VERSION="${1:-v1.2.1}"
OUTPUT_DIR="${2:-releases}"
APP_NAME="AstraTTS"
BUNDLE_ID="com.astratts.app"

mkdir -p "$OUTPUT_DIR"

# 确保所需命令存在
if ! command -v hdiutil &> /dev/null; then
    echo "❌ 当前系统不支持 hdiutil（仅 macOS 可生成 .dmg）。请在 macOS 主机上运行此脚本。"
    exit 1
fi

build_app_for_arch() {
    local arch="$1"        # arm64 / x64
    local pub_dir="publish-mac-$arch"
    if [ ! -d "$pub_dir" ]; then
        return 1
    fi

    echo ""
    echo "🍎 正在构建 $APP_NAME.app ($arch) ..."

    local stage_dir
    stage_dir=$(mktemp -d -t astratts_dmg_XXXXXX)
    local app_root="$stage_dir/$APP_NAME.app"
    local contents="$app_root/Contents"
    local macos_dir="$contents/MacOS"
    local resources_dir="$contents/Resources"

    mkdir -p "$macos_dir" "$resources_dir"

    # 1. 复制整个发布目录到 Resources/app (包含 astra-server, astra-cli, resources, tools, config 等)
    mkdir -p "$resources_dir/app"
    # 使用 ditto 保留权限与符号链接
    ditto "$pub_dir/" "$resources_dir/app/"

    # 确保可执行权限
    chmod +x "$resources_dir/app/astra-server" 2>/dev/null || true
    chmod +x "$resources_dir/app/astra-cli" 2>/dev/null || true
    chmod +x "$resources_dir/app/init-env-mac.sh" 2>/dev/null || true

    # 2. 启动器脚本：双击 .app 时启动 astra-server 并打开浏览器
    #    关键：启动器必须保持前台运行（exec 替换为 astra-server 进程），
    #    这样 macOS 才会在程序坞显示图标，并允许通过常规方式（Cmd+Q / 程序坞右键退出）关闭。
    cat > "$macos_dir/$APP_NAME" <<'LAUNCHER_EOF'
#!/bin/bash
# AstraTTS .app 启动器
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
APP_RES="$DIR/../Resources/app"

cd "$APP_RES"

# 解除 quarantine（首次运行后无害）
xattr -dr com.apple.quarantine "$APP_RES" 2>/dev/null || true

# 日志目录
LOG_DIR="$HOME/Library/Logs/AstraTTS"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/astra-server.log"

# 若已有同名服务在跑，直接打开浏览器后退出（避免端口冲突）
if pgrep -f "$APP_RES/astra-server" >/dev/null 2>&1; then
    echo "$(date) AstraTTS 已经在运行，仅打开浏览器" >> "$LOG_FILE"
    PORT="${ASTRATTS_PORT:-5000}"
    open "http://localhost:$PORT/" || true
    exit 0
fi

# 后台异步等待端口可用后再打开浏览器（不阻塞前台进程）
PORT="${ASTRATTS_PORT:-5000}"
(
    for i in {1..60}; do
        if curl -s "http://localhost:$PORT/" >/dev/null 2>&1; then
            open "http://localhost:$PORT/" || true
            exit 0
        fi
        sleep 1
    done
) &

# 关键：使用 exec 把当前 shell 进程替换为 astra-server，
# 这样 .app 的主进程（Contents/MacOS/AstraTTS）就持续存在，
# Dock 中会显示图标，Cmd+Q / 程序坞右键“退出”都能正常关闭进程。
exec "$APP_RES/astra-server" >> "$LOG_FILE" 2>&1
LAUNCHER_EOF
    chmod +x "$macos_dir/$APP_NAME"

    # 3. Info.plist
    #    LSUIElement=false / LSBackgroundOnly=false 显式声明这是常规 GUI 程序，
    #    确保进程出现在 Dock 中，并支持 Cmd+Q 退出。
    cat > "$contents/Info.plist" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>           <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>    <string>$APP_NAME</string>
    <key>CFBundleExecutable</key>     <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>     <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>        <string>${VERSION#v}</string>
    <key>CFBundleShortVersionString</key><string>${VERSION#v}</string>
    <key>CFBundlePackageType</key>    <string>APPL</string>
    <key>CFBundleSignature</key>      <string>????</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>LSMinimumSystemVersion</key> <string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSUIElement</key>            <false/>
    <key>LSBackgroundOnly</key>       <false/>
    <key>NSSupportsAutomaticTermination</key><false/>
    <key>NSSupportsSuddenTermination</key><false/>
</dict>
</plist>
PLIST_EOF

    # 4. 应用图标 (若存在)
    if [ -f "images/logo.png" ]; then
        if command -v sips &> /dev/null && command -v iconutil &> /dev/null; then
            local iconset="$stage_dir/AppIcon.iconset"
            mkdir -p "$iconset"
            for sz in 16 32 64 128 256 512; do
                sips -z $sz $sz images/logo.png --out "$iconset/icon_${sz}x${sz}.png" >/dev/null 2>&1 || true
                local d=$((sz * 2))
                sips -z $d $d images/logo.png --out "$iconset/icon_${sz}x${sz}@2x.png" >/dev/null 2>&1 || true
            done
            iconutil -c icns "$iconset" -o "$resources_dir/AppIcon.icns" 2>/dev/null || true
            if [ -f "$resources_dir/AppIcon.icns" ]; then
                /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string AppIcon" "$contents/Info.plist" 2>/dev/null || true
            fi
        fi
    fi

    # 5. 剥离 quarantine 属性
    xattr -dr com.apple.quarantine "$app_root" 2>/dev/null || true

    # 6. 制作 DMG: 使用一个临时 staging 文件夹包含 .app 与 Applications 软链
    local dmg_stage="$stage_dir/dmg"
    mkdir -p "$dmg_stage"
    ditto "$app_root" "$dmg_stage/$APP_NAME.app"
    ln -s /Applications "$dmg_stage/Applications"

    local dmg_path="$OUTPUT_DIR/AstraTTS-$VERSION-macOS-$arch.dmg"
    rm -f "$dmg_path"

    echo "🛠  正在生成 DMG -> $dmg_path"
    hdiutil create \
        -volname "$APP_NAME $VERSION" \
        -srcfolder "$dmg_stage" \
        -ov -format UDZO \
        "$dmg_path" >/dev/null

    echo "✅ DMG 已生成: $dmg_path"

    # 清理临时目录
    rm -rf "$stage_dir"
}

BUILT_ANY=0
if build_app_for_arch "arm64"; then BUILT_ANY=1; fi
if build_app_for_arch "x64"; then BUILT_ANY=1; fi

if [ $BUILT_ANY -eq 0 ]; then
    echo "❌ 未找到 publish-mac-arm64 或 publish-mac-x64 目录。请先运行 ./publish-mac.sh"
    exit 1
fi

echo ""
echo "🎉 所有 DMG 打包完成！输出目录: $(cd "$OUTPUT_DIR" && pwd)"