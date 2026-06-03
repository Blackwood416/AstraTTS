# AstraTTS macOS 兼容性改动文档

> 本文档详细记录此次为兼容 macOS 平台所做的全部改动。改动覆盖 **暂存区（staged）** 与 **工作区已修改未暂存（unstaged）** 共 **13 个文件**：3 个新增脚本 + 10 个修改文件。

## 一、改动总览

| 序号 | 文件 | 状态 | 行数变化 | 改动主题 |
| --- | --- | --- | --- | --- |
| 1 | [`init-env-mac.sh`](init-env-mac.sh) | 新增（暂存） | +67 | macOS 环境初始化（ffmpeg / Python venv / 依赖） |
| 2 | [`publish-mac.sh`](publish-mac.sh) | 新增（暂存） | +114 | macOS 双架构发布脚本 |
| 3 | [`pack-mac-dmg.sh`](pack-mac-dmg.sh) | 新增（暂存） | +183 | `.app` Bundle + `.dmg` 打包 |
| 4 | [`.gitignore`](.gitignore) | 修改 | +3 | 忽略 macOS 产物目录 |
| 5 | [`AstraTTS.CLI/AstraTTS.CLI.csproj`](AstraTTS.CLI/AstraTTS.CLI.csproj) | 修改 | +2 | 关闭 CA1416 跨平台 API 误报 |
| 6 | [`AstraTTS.Web/AstraTTS.Web.csproj`](AstraTTS.Web/AstraTTS.Web.csproj) | 修改 | +2 | 关闭 CA1416 跨平台 API 误报 |
| 7 | [`AstraTTS.Core/Config/TTSConfig.cs`](AstraTTS.Core/Config/TTSConfig.cs) | 修改 | +19 | 新增 `UseCoreML`、`CoreMLFlags` 配置项 |
| 8 | [`config.template.yaml`](config.template.yaml) | 修改 | +10 | 暴露 `UseCoreML` / `CoreMLFlags` 等配置 |
| 9 | [`AstraTTS.Core/Core/InferenceEngineV1.cs`](AstraTTS.Core/Core/InferenceEngineV1.cs) | 修改 | +110 / -16 | CoreML EP 支持 + 失败回退 + P-Core 探测 |
| 10 | [`AstraTTS.Core/Core/InferenceEngineV2.cs`](AstraTTS.Core/Core/InferenceEngineV2.cs) | 修改 | +51 / -1 | V2 引擎的 CoreML EP 支持 + spinning 切换 |
| 11 | [`AstraTTS.CLI/Program.cs`](AstraTTS.CLI/Program.cs) | 修改 | +106 / -25 | `LinuxAudioPlayer` → `UnixAudioPlayer`，新增 `afplay` 文件回退 |
| 12 | [`pack-release.sh`](pack-release.sh) | 修改 | +56 / -12 | 整合 macOS `.tar.gz` / `.dmg` 打包 |
| 13 | [`README.md`](README.md) | 修改 | +28 / -2 | 新增 macOS 安装、构建与依赖说明 |

合计：**13 个文件、约 +751 / -56 行**

---

## 二、新增脚本（暂存区）

### 2.1 [`init-env-mac.sh`](init-env-mac.sh)

面向最终用户在 macOS 上首次使用 AstraTTS 时自动准备环境。

- 检测 Homebrew（缺失仅提示，不强制安装）。
- 检查 `ffplay`（来自 ffmpeg），缺失且有 brew 时自动 `brew install ffmpeg`；否则降级到 `afplay`。
- 校验 `python3` 存在（缺失则报错并提示 `brew install python`）。
- 在 `tools/converter/.venv` 创建 Python 虚拟环境，升级 `pip`，安装 `onnx`、`numpy`、`onnxsim`、`onnxruntime` 与 CPU 版 `torch`。
- 使用 `set -e` 严格失败终止；与 Windows 版 `init-env.*` 完全解耦。

### 2.2 [`publish-mac.sh`](publish-mac.sh)

macOS 双架构发布脚本。

- 通过参数或 `uname -m` 解析 `RID` 为 `osx-arm64` / `osx-x64`，输出至 `publish-mac-<arch>`。
- 校验 `dotnet`，清理旧目录后分别 `dotnet publish` [`AstraTTS.Web`](AstraTTS.Web/AstraTTS.Web.csproj)（产物 `astra-server`）与 [`AstraTTS.CLI`](AstraTTS.CLI/AstraTTS.CLI.csproj)（产物 `astra-cli`），开关 `--self-contained true`、`PublishSingleFile=false`、`AllowMissingPrunePackageData=true`。
- 复制 [`config.template.yaml`](config.template.yaml) → `config.template.yaml` 与 `config.yaml`；如有 `resources-minimal/` 复制为 `resources/`；复制 `tools/converter/v1_converter.py` 与 `templates/`，并显式删除 Windows 专用 `tools/converter/runtime/`。
- 拷贝 [`init-env-mac.sh`](init-env-mac.sh) 进入产物，赋可执行权限；`xattr -dr com.apple.quarantine` 剥离 Gatekeeper 隔离。

### 2.3 [`pack-mac-dmg.sh`](pack-mac-dmg.sh)

将发布产物组装成 `.app` Bundle 并制作 `.dmg`。

- 校验 `hdiutil`；遍历 `publish-mac-arm64`、`publish-mac-x64` 两个架构。
- 用 `ditto` 把整个发布目录复制到 `Contents/Resources/app/`，确保权限与软链保留。
- 生成 `Contents/MacOS/AstraTTS` 启动器：自动 `xattr` 解锁、日志写 `~/Library/Logs/AstraTTS/`、`pgrep` 防重复启动、后台轮询端口 60s 后自动 `open` 浏览器；**关键使用 `exec` 把 shell 替换为 `astra-server` 主进程**，确保 Dock 图标显示与 `Cmd+Q` 退出可用。
- 写入 `Info.plist`：`CFBundleIdentifier=com.astratts.app`、`LSMinimumSystemVersion=11.0`、`NSHighResolutionCapable=true`、显式声明 `LSUIElement=false` / `LSBackgroundOnly=false` / `NSSupportsAutomaticTermination=false`。
- 若 [`images/logo.png`](images/logo.png) 存在，使用 `sips` + `iconutil` 自动生成 16~512 全尺寸（含 @2x）`AppIcon.icns` 并由 `PlistBuddy` 注入。
- `hdiutil create ... -format UDZO` 输出 `releases/AstraTTS-<VERSION>-macOS-<arch>.dmg`，DMG 内含 `Applications` 软链以提示拖入安装。

---

## 三、工程配置改动

### 3.1 [`.gitignore`](.gitignore)

```diff
+/releases/
+/publish-mac-arm64/
```

避免 macOS 产物入库。

### 3.2 [`AstraTTS.CLI/AstraTTS.CLI.csproj`](AstraTTS.CLI/AstraTTS.CLI.csproj) 与 [`AstraTTS.Web/AstraTTS.Web.csproj`](AstraTTS.Web/AstraTTS.Web.csproj)

均新增：

```xml
<NoWarn>$(NoWarn);CA1416</NoWarn>
```

跨平台发布时 NAudio `WasapiOut` 等 Windows 专用 API 已通过运行时分支隔离，关闭编译器 CA1416 误报，保证 Linux/macOS publish 不被告警淹没。

### 3.3 [`AstraTTS.Core/Config/TTSConfig.cs`](AstraTTS.Core/Config/TTSConfig.cs)

新增两个配置项（仅 macOS 生效）：

- `bool UseCoreML`（默认 `false`）：是否启用 CoreML EP；默认关闭，原因：实测 GPT-SoVITS 这类含循环控制流的模型 CoreML 子图切分反而比纯 CPU 更慢，且首次加载需要 ANE 编译数十秒。
- `uint CoreMLFlags`（默认 `2`）：标志位组合，按位或：
  - `0x2` `EnableOnSubgraphs`（子图回退）— 默认推荐
  - `0x4` `OnlyEnableDeviceWithANE`（优先 ANE）
  - `0x8` `OnlyAllowStaticInputShapes`
  - `0x10` `CreateMLProgram`（MLProgram 格式）

### 3.4 [`config.template.yaml`](config.template.yaml)

为最终用户暴露上述配置项，并补充注释：

- `IntraOpNumThreads` 注释说明 macOS 自动模式倾向于选择 P-Core 数。
- 新增 `UseDirectML`（Windows 专用）、`UseCoreML`、`CoreMLFlags`，并在注释中明确 macOS 默认关闭 CoreML 的理由。

---

## 四、推理引擎改动

### 4.1 [`AstraTTS.Core/Core/InferenceEngineV1.cs`](AstraTTS.Core/Core/InferenceEngineV1.cs)

主要变化：

1. **`GetSessionOptions` 重载**：新增 `GetSessionOptions(config, bool useCoreML)`，便于失败回退时单独构造无 CoreML 的 `SessionOptions`。
2. **CPU 自旋策略平台化**：`session.intra_op.allow_spinning` 在 macOS 设为 `"1"`、其他平台保持 `"0"`，原因是 mac 调度延迟较大，自旋反而能降低首块时延。
3. **CoreML EP 接入**：`isMac && useCoreML` 时关闭 `MemoryPattern` 并 `AppendExecutionProvider("CoreML", options)`，将 `CoreMLFlags` 翻译为 `EnableOnSubgraphs` / `MLComputeUnits` / `ModelFormat` / `RequireStaticInputShapes`，包裹 `try/catch` 异常即降级。
4. **新增 `LoadModelWithFallback`**：每个模型独立 try CoreML，失败则按模型粒度回退到 CPU，不至于让一份模型的兼容问题阻断整个引擎。
5. **`LoadModels` 全部改用 `LoadModelWithFallback`** 加载 `t2s_encoder` / `t2s_first_stage_decoder` / `t2s_stage_decoder` / `vits` / `prompt_encoder` / `hubert` / `sv` 七个模型。
6. **新增 `GetMacPerformanceCoreCount` / `ReadSysctlInt`** 工具：通过 `sysctl -n hw.perflevel0.physicalcpu` 查询 Apple Silicon P-Core 数（M1/M2 为 4，M2 Pro 为 6/8 等），失败回退 `hw.physicalcpu` 或 `Environment.ProcessorCount/2`。当前用于配置参考与诊断（线程数仍以用户 config 为准）。

### 4.2 [`AstraTTS.Core/Core/InferenceEngineV2.cs`](AstraTTS.Core/Core/InferenceEngineV2.cs)

V2 引擎对齐 V1 的平台化处理：

- 同样新增 `isMac` 判断与 `allow_spinning` 在 mac 上设为 `"1"`。
- 在非 Windows、`isMac && config.UseCoreML` 分支添加 CoreML EP 配置（与 V1 同样的标志位翻译表），异常时回退 CPU 并恢复 `EnableMemoryPattern`。
- 新增本地版 `GetMacPerformanceCoreCount`。

> 注：当前 macOS 发布默认仅启用 V1 引擎，V2 / v2ProPlus 在 mac 上的端到端验证仍在进行中（README 已注明）。

---

## 五、CLI 跨平台播放器重构

### 5.1 [`AstraTTS.CLI/Program.cs`](AstraTTS.CLI/Program.cs)

将原 `LinuxAudioPlayer` **重命名并扩展为 `UnixAudioPlayer`**，覆盖 Linux 与 macOS：

- **平台探测**：通过 `RuntimeInformation.IsOSPlatform(OSPlatform.OSX)` 判定 mac。
- **播放器优先级调整**：优先 `ffplay`、`play`(sox)，其次平台专用 `pw-play` / `paplay` / `aplay`（仅 Linux），mac 末位回退 `afplay`。
- **新增 `afplay` 文件回退模式**：因 `afplay` 不支持 `stdin`，将整段 PCM 累积进 `MemoryStream`，在 `WaitForFinish` 时写出临时 `WAV` 文件再 `afplay` 阻塞播放：
  - 新增字段 `_isMac` / `_pcmBuffer` / `_sampleRate` / `_bufLock` / `_tempWavPath`；
  - 新增 `WriteWavFile` 方法手写 RIFF / WAVE / fmt / data 块；
  - `Dispose` 时清理临时 `astratts_stream_*.wav`。
- 调整 `ffplay` 参数：增加 `-loglevel quiet`；`aplay` 增加 `-q`；`paplay` 改用 `--raw`；进程退出超时 2s → 5s。
- 调用方由 `LinuxAudioPlayer` 全部更名为 `UnixAudioPlayer`（共 8 处替换：`linuxPlayer` → `unixPlayer`）。

---

## 六、打包与文档

### 6.1 [`pack-release.sh`](pack-release.sh)

- 文件模式：`100644` → `100755`（变成可执行）。
- 头注释：`(Linux)` 改为 `(Linux & macOS)`。
- Linux 打包改为子 shell 形式 `(cd publish-linux && tar -czf ...)` 并去掉冗余 `-v`，避免污染父目录 `pwd`。
- 新增 `pack_mac` 函数：遍历 `publish-mac-arm64` / `publish-mac-x64`，复制 `init-env-mac.sh`、生成 `AstraTTS-<VER>-macOS-<arch>.tar.gz`。
- 自动调用 [`pack-mac-dmg.sh`](pack-mac-dmg.sh) 生成 `.dmg`（如发布目录存在）。
- 新增独立资源包逻辑：将 `resources-minimal/` 打包为 `AstraTTS-resources-minimal-<VER>.tar.gz`。
- 输出路径改为 `cd $RELEASE_DIR && pwd`（兼容 macOS 缺失 `realpath` 的情况）。

### 6.2 [`README.md`](README.md)

- 顶部多平台说明追加 macOS（Apple Silicon / Intel）。
- 章节标题调整：`整合包极速体验 (Windows / Linux)` → `(Windows / Linux / macOS)`。
- 新增 macOS 用户安装小节，覆盖 **DMG 安装包**（拖入 Applications + 首次右键打开绕 Gatekeeper）和 **tar.gz 整合包**（解压、`xattr -dr com.apple.quarantine`、`./init-env-mac.sh`、`./astra-server` / `./astra-cli`）两种方式；明确"仅启用 v1 推理引擎"的限制。
- 系统要求章节追加 macOS 11+ 双架构、`brew install ffmpeg` 推荐播音；构建与发布章节追加 `publish-mac.sh`、`pack-release.sh`、`pack-mac-dmg.sh`。

---

## 七、整体使用流程

```bash
# 1. 开发机一次性环境准备
./init-env-mac.sh

# 2. 发布二进制（任选其一或全部架构）
./publish-mac.sh arm64
./publish-mac.sh x64

# 3a. 打包 .tar.gz + .dmg（全平台一站式）
./pack-release.sh

# 3b. 仅打包 .dmg
./pack-mac-dmg.sh v1.2.1 releases
```

最终在 [`releases/`](releases/) 下产出 `AstraTTS-v1.2.1-macOS-arm64.dmg` / `.tar.gz` 等。

---

## 八、设计原则与注意事项

1. **跨平台对称改造**：
   - `LinuxAudioPlayer` → `UnixAudioPlayer`（语义扩展）；
   - `GetSessionOptions` 引入 `isMac` 分支，但保留 Win/Linux 既有路径不变；
   - 增加构件层 `LoadModelWithFallback` 包裹模型加载，最大化 mac 兼容性而不影响其他平台。
2. **CoreML 默认关闭**：实测对 GPT-SoVITS 类带控制流模型 CoreML 反而劣化，且 ANE 首次编译耗时数十秒。开关与标志位都暴露给用户调优。
3. **Gatekeeper 三道防线**：发布脚本、打包脚本、`.app` 启动器都执行 `xattr -dr com.apple.quarantine`。
4. **零侵入既有 Windows/Linux 流程**：所有 macOS 路径要么在新文件中，要么在现有文件里以平台判断分支隔离。
5. **未签名说明**：脚本未集成 `codesign` / `notarytool`，最终用户首次运行需要右键打开。如需上架或避免 "无法验证开发者" 警告，可在 [`pack-mac-dmg.sh`](pack-mac-dmg.sh) 末尾追加签名 / 公证步骤。
6. **架构无关性**：`arm64` 与 `x64` 构建逻辑完全对称、相互独立。
