# AstraTTS

<p align="center">
  <img src="images/banner.png" alt="AstraTTS Banner" width="400">
</p>

<p align="center">
  <strong>🎙️ 高性能跨平台 TTS (Text-to-Speech) 引擎</strong>
</p>

<p align="center">
  基于 ONNX Runtime 的高性能语音合成解决方案，支持流式输出、高性能并发推理、多音色管理。
</p>

<p align="center">
  <a href="./README_EN.md">English</a> | 简体中文
</p>

<p align="center">
  <a href="https://jq.qq.com/?_wv=1027&k=yvN60wYc"><img src="https://img.shields.io/badge/QQ%E7%BE%A4-1083409411-blue?style=flat-square&logo=qq"></a>
</p>

> [!IMPORTANT]
> **重要说明**：由于核心模型体积较大且 Git LFS 开销较高，项目现已**停止使用 Git LFS**。核心资源（`resources-minimal`）现已移至 [GitHub Releases](https://github.com/Blackwood416/AstraTTS/releases) 独立托管。**从源码构建或使用 Docker 的用户，请务必手动下载资源包并解压至项目根目录。**

---

## 📢 新版本 v1.2.1 发布！

AstraTTS v1.2.1 带来了全面的部署优化和体验升级，特别是针对 Linux 与容器化环境的支持。

### ✨ 新增功能与改进
- 🐳 **原生 Docker 支持** - 提供极简优化的 Dockerfile 部署，基于 `.NET 10 (Ubuntu Noble)` 和原生 Python 环境构建。一键启动，免除环境配置烦恼，针对国内网络环境已默认启用清华源加速。
- 🔄 **WebUI 快速重置** - Web 管理面板新增一键重置功能，方便用户随时初始化或恢复打乱的全局配置，优化调试体验。
- 🐧 **Linux 音频适配** - Linux 环境下自带 CLI 的音频播放组件实现了智能静默认降级处理（`pw-play` -> `paplay` -> `aplay`），提升不同发行版兼容性。
- 🚀 **全面支持 v2ProPlus 及并发增强** - 继续保留并优化对于 GPT-SoVITS V2ProPlus 架构模型的并行加载能力及并发合成处理。

---

## ✨ 项目特性

- 🚀 **高性能推理** - 基于 ONNX Runtime，针对 CPU 指令集深度优化，推理速度远快于传统的 Python 推理。
- ⚖️ **高并发支持** - 内置推理池设计，支持多路并发合成，充分利用多核 CPU 资源。
- 🎵 **流式输出** - 毫秒级首包延迟，支持边合成边播放，告别卡顿。
- 🎭 **可视化管理** - 强大的 WebUI 面板，涵盖音色管理、模型转换与参数调节。

<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
  <img src="images/webui_light_theme.png" alt="WebUI Light Theme" style="width: 30%; margin: 10px;">
  <img src="images/voice_manager_light_theme.png" alt="Voice Manager Light Theme" style="width: 30%; margin: 10px;">
  <img src="images/webui_dark_theme.png" alt="WebUI Dark Theme" style="width: 30%; margin: 10px;">
  <img src="images/model_converter_dark_theme.png" alt="Model Converter" style="width: 30%; margin: 10px;">
</div>

- 🐧 **多平台支持** - 原生支持 Windows 10/11，并已实现 Linux (如 Ubuntu, Arch, WSL) 的完整兼容。
- 🌐 **多语言支持** - 完善的中/英、中/日双语混读支持（三语混合尚在开发中）。
- 🔄 **热重载** - 配置项可以在服务运行时即刻生效。

---

## 📦 安装与部署指南

### 1. 整合包极速体验 (Windows / Linux)
推荐国内用户通过 **夸克网盘** 下载完整的环境整合包（内含所有运行环境与默认模型）。
- **获取链接**: [https://pan.quark.cn/s/416fa9f65f3b](https://pan.quark.cn/s/416fa9f65f3b)
- **提取码**: `y8Wx`

#### 对于 Windows 用户 (`-win64.zip`)：
1. **下载并解压** 整合包。
2. **启动 Web 控制面板**: 运行 `astra-server.exe`。
3. **访问界面**: 浏览器打开 `http://localhost:5000`。
   - 在“模型转换”页可导入 SoVITS 模型。
   - 在“音色库管理”页可上传参考音频并调整参数。
4. **命令行单次播音**: 运行 `astra-cli.exe` 进行本地低延迟朗读。

#### 对于 Linux 用户 (`-linux64.tar.gz`)：
1. **下载并直接解压**: `tar -xzvf AstraTTS-v*-linux64.tar.gz`
2. **初始化模型转换环境**: 
   - Linux 环境由于未内嵌数百 MB 的 Python 构建环境，需要手动初始化依赖。
   - 在解压后的根目录运行：`./init-env.sh` (此脚本将在 `tools/converter/.venv` 下自动搭建所需的轻量级虚拟环境)。
3. **启动引擎**:
   - `chmod +x astra-server && ./astra-server` 即可启动 Web 服务。
   - 本地体验可以同样执行 `./astra-cli --text "测试"`。

### 2. Docker 部署 (推荐服务器使用)
项目内置了由多阶段构建优化的 Dockerfile。你可以通过下载模型独立包 (`resources-minimal`) 来部署：

- **模型资源获取**: 由于核心模型体积较大，项目不再通过 Git LFS 托管。请前往 [GitHub Releases](https://github.com/Blackwood416/AstraTTS/releases) 下载 `resources-minimal.zip` ，并将其解压到源码根目录下的 `resources-minimal` 文件夹中。

```bash
# 1. 克隆代码仓库
git clone https://github.com/Blackwood416/AstraTTS.git
cd AstraTTS

# 2. 准备模型资源
# 从 GitHub Releases 快速下载并解压核心资源
wget https://github.com/Blackwood416/AstraTTS/releases/latest/download/resources-minimal.zip
unzip resources-minimal.zip
rm resources-minimal.zip

# 3. 极速构建 Docker 镜像 (已针对国内网络加速，依赖的 apt 与 pip 已默认使用清华源，docker 镜像文件使用渡渡鸟镜像同步站 https://docker.aityp.com/ 提供的加速节点，如果发现节点无法访问可以自行修改 Dockerfile 中的镜像源)
docker build -t astratts-server:latest .

# 4. 运行容器
# 建议通过 -v 参数将本地的 resources 目录挂载进容器内部，以便于你随时在宿主机管理和更新模型文件。
docker run -d --name astratts \
  -p 5000:5000 \
  -v ./resources:/app/resources \
  astratts-server:latest
```

容器启动后，在浏览器访问 `http://localhost:5000` 即可见到完整的 Web 控制面板。所有的配置热重载、音色库管理以及模型转换功能均可完美在容器中工作。

### 3. 局域网访问 (跨设备使用)

如果你想在同一局域网下的其他手机或电脑上访问 Web 控制面板，可以使用 ASP.NET 自带的命令行参数 `--urls` 来指定监听所有网络接口：

- **Windows**: 打开终端（PowerShell 或 CMD），运行 `.\astra-server.exe --urls "http://0.0.0.0:5000"`
- **Linux**: 运行 `./astra-server --urls "http://0.0.0.0:5000"`

启动后，在其他设备的浏览器中输入这台电脑的局域网 IP 地址即可访问（例如 `http://192.168.1.100:5000`）。

---

## 📦 项目结构

- **AstraTTS.Core**: 核心 SDK，包含混合 G2P 引擎、RoBERTa/Hubert 特征提取器及高性能各版本推理引擎。
- **AstraTTS.CLI**: 命令行交互工具。Windows 下支持低延迟 WASAPI 播放；Linux 下通过管道适配 `aplay`/`paplay`/`pw-play` 等音频后端。
- **AstraTTS.Web**: 后端 Web 服务，集成全套 WebUI 管理功能，支持在 Linux 服务器 headless 部署。

---

## 🔧 引擎版本对比

AstraTTS 默认使用 **V1 引擎 (推荐)**。V2 目前仍处于较低成熟阶段。

| 特性 | V1 引擎 (推荐) | V2 引擎 (实验性) |
| :--- | :--- | :--- |
| **项目来源** | 基于 [Genie-TTS](https://github.com/High-Logic/Genie-TTS) | 基于 [GPT-SoVITS-Minimal](https://github.com/GPT-SoVITS_minimal) |
| **状态** | ✅ 稳定，支持 **V2ProPlus** 模型克隆 | 🚧 开发中 (WIP) |
| **语种能力** | ✅ 中日混读 / 中英混读 (双语) | ⚠️ 仅中英 |
| **采样参数** | ❌ 确定性生成 (不支持 TopK/Temp) | ✅ 支持 TopK / Temp / NoiseScale |
| **并发能力** | ✅ 支持 (Inference Pool) | ✅ 支持 |

---

#### 📂 模型与资源目录

资源结构（1.1.x+ 版本）：

```text
resources/
├── models_v1/                   # V1 引擎模型 (扁平化)
│   └── {avatarId}/              # 核心 vits.onnx 存放处
├── models_v2/                   # V2 引擎模型
│   └── {avatarId}/              # sovits.onnx 等
├── shared/                      # 基础库
│   ├── g2p/                     # 日语/中/英字典及模型
│   ├── v1_extra/                # 通用 Bert/Hubert 组件
├── avatars/                     # 角色音色库
│   └── {avatarId}/              # 参考音频目录
```

---


## ⚙️ 配置文件说明 (`config.yaml`)

新版本全面转向 YAML 格式。以下为核心配置项示例（完整说明见 `config.template.yaml`）：

```yaml
ResourcesDir: "resources"      # 资源位置
DefaultAvatarId: "default"     # 默认音色

# 性能配置
IntraOpNumThreads: 0           # 0 为自动线程数
InterOpNumThreads: 1           # 算子间并行

# 引擎与推理
UseEngineV2: false             # 默认使用 V1 
Speed: 1.0                     # 默认语速
StreamingMode: true            # 开启流式

# 音色定义
Avatars:
- Id: default
  Name: "我的音色"
  References:
  - Id: normal
    AudioPath: "ref.wav"
    Language: "zh"             # 指定参考音频语种
```

---

## 🚀 开发者快速开始 (从源码构建)

### 运行环境
- .NET 10.0 SDK。
- **Windows**: 10/11 (x64/arm64)。
- **Linux**: 支持基于 x86_64 或 ARM64 的主流发行版（如 Ubuntu 22.04+, Arch Linux, WSL2）。
  - **Linux 依赖**: 需要安装 `dotnet-runtime-10.0`。
  - **Linux 播音**: CLI 模式下推荐安装 `alsa-utils` (`aplay`) 或 `pulseaudio` (`paplay`)。

### 构建与发布
- **Windows**: 运行 `publish.ps1`。
- **Linux**: 运行 `publish-linux.sh`。

## 📄 许可证
MIT License

## 🙏 致谢
- [Genie-TTS](https://github.com/High-Logic/Genie-TTS) - V1 推理引擎核心架构参考。
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - V2 引擎算法来源。
- [GPT-SoVITS Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) - V2 C# 推理实现参考。
- [ONNX Runtime](https://onnxruntime.ai/) - 高性能跨平台推理后端。
- [NAudio](https://github.com/naudio/NAudio) - .NET 音频处理。
- [wasapi_relink](https://github.com/Litttlefish/wasapi_relink) - WASAPI 低延迟优化辅助组件。
- [BreakingBad (AI-Hobbyist)](https://www.ai-hobbyist.com/thread-1143-1-1.html) - 整合包内置默认模型来源。
