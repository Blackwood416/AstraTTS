# AstraTTS

<p align="center">
  <img src="images/banner.png" alt="AstraTTS Banner" width="800">
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

---

## 📢 新版本 v1.2.1 发布！

AstraTTS v1.2.1 带来了全面的部署优化和体验升级，特别是针对 Linux 与容器化环境的支持。

### ✨ 新增功能与改进
- 🐳 **原生 Docker 支持** - 提供极简优化的 Dockerfile 部署，基于 `.NET 10 (Ubuntu Noble)` 和原生 Python 环境构建。一键启动，免除环境配置烦恼，针对国内网络环境已默认启用清华源加速。
- 🗃️ **Git LFS 资源托管** - `resources-minimal` （核心模型、词典）及 `tools`（模型转换器脚本）现已完全并入 Git LFS 进行版本控制。 `git lfs pull` 即可无缝获取运行所需的所有完整组件，告别手动下载配置压缩包。
- 🔄 **WebUI 快速重置** - Web 管理面板新增一键重置功能，方便用户随时初始化或恢复打乱的全局配置，优化调试体验。
- 🐧 **Linux 音频适配** - Linux 环境下自带 CLI 的音频播放组件实现了智能静默认降级处理（`pw-play` -> `paplay` -> `aplay`），提升不同发行版兼容性。
- 🚀 **全面支持 v2ProPlus 及并发增强** - 继续保留并优化对于 GPT-SoVITS V2ProPlus 架构模型的并行加载能力及并发合成处理。

---

## ✨ 项目特性

- 🚀 **高性能推理** - 基于 ONNX Runtime，针对 CPU 指令集深度优化，推理速度远快于传统的 Python 推理。
- ⚖️ **高并发支持** - 内置推理池设计，支持多路并发合成，充分利用多核 CPU 资源。
- 🎵 **流式输出** - 毫秒级首包延迟，支持边合成边播放，告别卡顿。
- 🎭 **可视化管理** - 强大的 WebUI 面板，涵盖音色管理、模型转换与参数调节。
- 🐧 **多平台支持** - 原生支持 Windows 10/11，并已实现 Linux (如 Ubuntu, Arch, WSL) 的完整兼容。
- 🌐 **多语言支持** - 完善的中/英、中/日双语混读支持（三语混合尚在开发中）。
- 🔄 **热重载** - 配置项可以在服务运行时即刻生效。

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

## 📦 整合包使用说明

1.  **下载并解压** 整合包。
2.  **启动 Web 控制面板**: 运行 `astra-server.exe`。
3.  **访问界面**: 浏览器打开 `http://localhost:5000`。
    - 在“模型转换”页可导入 SoVITS 模型。
    - 在“音色库管理”页可上传参考音频并调整参数。
4.  **命令行使用**: 运行 `astra-cli.exe` 进行本地低延迟朗读。

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

### Docker 部署 (推荐服务器使用)
项目内置了 Dockerfile，支持快速容器化部署：
```bash
# 1. 构建镜像
docker build -t astratts-server .

# 2. 运行容器 (挂载宿主机的 resources 目录以持久化模型和配置)
docker run -d -p 5000:5000 -v ./resources:/app/resources astratts-server
```
启动后即可访问 `http://localhost:5000`。

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
