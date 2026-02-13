# AstraTTS

<p align="center">
  <img src="images/banner.png" alt="AstraTTS Banner" width="800">
</p>

<p align="center">
  <strong>🎙️ 高性能 Windows 专用 TTS (Text-to-Speech) 引擎</strong>
</p>

<p align="center">
  基于 ONNX Runtime 的高性能语音合成解决方案，支持流式输出、高性能并发推理、多音色管理。
</p>

<p align="center">
  <a href="./README_EN.md">English</a> | 简体中文
</p>

---

## 📢 新版本 v1.1.1 测试发布！

AstraTTS 迎来重大更新，正式引入日语支持与全新的 Web 可视化管理界面。

### ✨ 新增功能与改进
- 🇯🇵 **日语支持** - 引入基于 OpenJTalk 的日语 G2P 系统，支持中日动态混读（目前仅支持双语混合，不支持三语同载）。
- 🖥️ **全新 WebUI** - 启动 `astra-server.exe` 后访问 `http://localhost:5000` 即可：
  - **音色管理**：可视化编辑、批量参考音频管理（支持直接上传音频文件）。
  - **配置编辑**：在线修改全局参数，支持热重载。
  - **在线调试**：流式/非流式合成测试、性能压力测试。
  - **模型转换**：集成针对 **GPT-SoVITS V2ProPlus** 架构优化的转换逻辑，支持网页端一键操作。
- ⚖️ **并发推理增强** - 引入推理引擎池（Inference Pool），利用多核 CPU 实现高并发请求处理，支持 `PoolCapacity` 配置。
- 🧠 **G2P 系统重构** - 全面重写字素到音素系统，显著提升处理速度与多音字纠准。
- 📝 **YAML 配置支持** - 统一使用 YAML 格式 (`config.yaml`)。
  - **结构清晰**：完美支持注释，由 `config.template.yaml` 提供详尽参数说明。
- 📂 **目录结构优化** - 扁平化 `models_v1` 存储结构，移除冗余嵌套，优化加载路径。

---

## ✨ 项目特性

- 🚀 **高性能推理** - 基于 ONNX Runtime，针对 CPU 指令集深度优化，推理速度远快于传统的 Python 推理。
- ⚖️ **高并发支持** - 内置推理池设计，支持多路并发合成，充分利用多核 CPU 资源。
- 🎵 **流式输出** - 毫秒级首包延迟，支持边合成边播放，告别卡顿。
- 🎭 **可视化管理** - 强大的 WebUI 面板，涵盖音色管理、模型转换与参数调节。
- 🌐 **多语言支持** - 完善的中/英、中/日双语混读支持（三语混合尚在开发中）。
- 🔄 **热重载** - 配置项可以在服务运行时即刻生效。

## 📦 项目结构

- **AstraTTS.Core**: 核心 SDK，包含混合 G2P 引擎、RoBERTa/Hubert 特征提取器及高性能各版本推理引擎。
- **AstraTTS.CLI**: Windows 专用命令行交互工具，支持低延迟 WASAPI 直接播放。
- **AstraTTS.Web**: 后端 Web 服务，集成全套 WebUI 管理功能。

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
- **Windows 10/11** (目前强制依赖 Windows，不支持跨平台)。

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
