# AstraTTS

<p align="center">
  <img src="images/banner.png" alt="AstraTTS Banner" width="800">
</p>

<p align="center">
  <strong>🎙️ High-Performance Windows-Only TTS (Text-to-Speech) Engine</strong>
</p>

<p align="center">
  A high-performance speech synthesis solution based on ONNX Runtime, supporting streaming output, concurrent inference, and multi-voice management.
</p>

<p align="center">
  English | <a href="./README.md">简体中文</a>
</p>

---

## 📢 New Version v1.1.1 Beta Released!

AstraTTS has received a major update, introducing Japanese support and a comprehensive Web-based management interface.

### ✨ New Features & Improvements
- 🇯🇵 **Japanese Support** - Integrated OpenJTalk-based Japanese G2P system, supporting dynamic mixed synthesis (Current version supports dual-language mixing e.g., ZH-JA/ZH-EN, triple-language mix is not yet supported).
- 🖥️ **All-New WebUI** - Access it at `http://localhost:5000` after starting `astra-server.exe`:
  - **Voice Management**: Visual editing and batch reference audio management (direct upload supported).
  - **Config Editor**: Online modification of global parameters with hot-reload support.
  - **Interactive Debugging**: Real-time synthesis tests and performance stress testing.
  - **Model Converter**: Integrated conversion logic optimized for **GPT-SoVITS V2ProPlus** architectures.
- ⚖️ **Concurrency Enhancement** - Introduced an **Inference Engine Pool** to handle multiple simultaneous requests efficiently using multi-core CPUs, configurable via `PoolCapacity`.
- 🧠 **G2P System Refactored** - Completely rewritten Graphene-to-Phoneme engine for significantly faster processing and improved polyphone accuracy.
- 📝 **YAML Configuration Support** - Standardized on YAML format (`config.yaml`).
  - **Readability**: Full comment support provided in `config.template.yaml` for all parameters.
- 📂 **Directory Structure Optimization** - Flattened the `models_v1` hierarchy, removing redundant subfolders for more efficient asset loading.
- 🔍 **Streamlined API Docs** - Optimized the built-in Scalar API page, hiding internal endpoints for a cleaner developer experience.

---

## ✨ Features

- 🚀 **High-Performance** - Powered by ONNX Runtime and optimized for CPU instruction sets, far exceeding traditional Python-based inference.
- ⚖️ **High Concurrency** - Built-in inference pool design allows simultaneous processing of multiple synthesis requests.
- 🎵 **Streaming Synthesis** - Millisecond-level first-chunk latency with "play-while-synthesizing" support.
- 🎭 **Visual Management** - Powerful WebUI dashboard for voice bank management, model conversion, and parameter tuning.
- 🔧 **Flexible Deployment** - Ideal for both lightweight CLI use and distributed Web API services.
- 🌐 **Mixed-Language Synthesis** - Robust support for ZH-EN and ZH-JA bilingual mixing (trilingual mix is still in development).
- 🔄 **Hot Reload** - Configuration changes take effect immediately without service interruption.

## 📦 Project Structure

- **AstraTTS.Core**: Core SDK containing hybrid G2P engines, RoBERTa/Hubert extractors, and high-performance inference modules.
- **AstraTTS.CLI**: Windows-exclusive command-line tool with low-latency WASAPI playback.
- **AstraTTS.Web**: Backend Web service providing RESTful APIs and the management dashboard.

---

## 🔧 Engine Versions

AstraTTS uses **V1 Engine (Recommended)** by default. V2 remains experimental.

| Feature | V1 Engine (Recommended) | V2 Engine (Experimental) |
| :--- | :--- | :--- |
| **Origin** | Based on [Genie-TTS](https://github.com/High-Logic/Genie-TTS) | Based on [GPT-SoVITS-Minimal](https://github.com/GPT-SoVITS_minimal) |
| **Status** | ✅ Stable, supports **V2ProPlus** cloning | 🚧 WIP, Experimental |
| **Language Mix** | ✅ ZH-JA / ZH-EN (Bilingual Only) | ⚠️ ZH-EN Only |
| **Sampling** | ❌ Deterministic (No TopK/Temp) | ✅ TopK / Temp / NoiseScale |
| **Concurrency** | ✅ Supported (Inference Pool) | ✅ Supported |

#### 📂 Models & Resources

Resource structure for version 1.1.x+:

```text
resources/
├── models_v1/                   # V1 Engines (Flattened)
│   └── {avatarId}/              # core vits.onnx is located here
├── models_v2/                   # V2 Engines
│   └── {avatarId}/              # sovits.onnx etc.
├── shared/                      # Base Resources
│   ├── g2p/                     # Dictionaries (JA/ZH/EN)
│   ├── v1_extra/                # Common Bert/Hubert components
├── avatars/                     # Voice Library
│   └── {avatarId}/              # Reference audio folder
```

---

## 📦 Portable Package Usage

1.  **Download and Extract** the integration package.
2.  **Start Web Dashboard**: Run `astra-server.exe`.
3.  **Access Interface**: Open `http://localhost:5000` in your browser.
    - Use the "Converter" tab to import your SoVITS models.
    - Use the "Avatars" tab to upload reference audios and tune parameters.
4.  **CLI Testing**: Run `astra-cli.exe` for local low-latency playback.

---

## ⚙️ Configuration Guide (`config.yaml`)

The system now prioritizes YAML format. Core configuration example (see `config.template.yaml` for full details):

```yaml
ResourcesDir: "resources"      # Resource path
DefaultAvatarId: "default"     # Default voice ID

# Performance Settings
IntraOpNumThreads: 0           # 0 for auto
InterOpNumThreads: 1           # Inter-operator parallelism
PoolCapacity: 4                # Inference engine pool size for concurrency

# Engine & Inference
UseEngineV2: false             # default to V1
Speed: 1.0                     # default speed
StreamingMode: true            # enable streaming

# Voice Definitions
Avatars:
- Id: default
  Name: "My Voice"
  References:
  - Id: normal
    AudioPath: "ref.wav"
    Language: "zh"             # Specify reference audio language
```

---

## 🚀 Developer Quick Start (Source Build)

### Prerequisites
- .NET 10.0 SDK.
- **Windows 10/11** (Windows is mandatory; other platforms are not supported).

### Build & Run
1.  Clone the repository: `git clone https://github.com/your-repo/AstraTTS.git`
2.  Navigate to the project directory: `cd AstraTTS`
3.  Restore dependencies: `dotnet restore`
4.  Build the solution: `dotnet build`
5.  Run the web server: `dotnet run --project AstraTTS.Web`
6.  Access the WebUI at `http://localhost:5000`.

---

## 📄 License
MIT License

## 🙏 Acknowledgments
- [Genie-TTS](https://github.com/High-Logic/Genie-TTS) - Core architecture reference for V1 engine.
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - Algorithm source for V2 engine.
- [GPT-SoVITS Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) - C# inference reference for V2.
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance cross-platform backend.
- [NAudio](https://github.com/naudio/NAudio) - .NET Audio processing.
- [wasapi_relink](https://github.com/Litttlefish/wasapi_relink) - WASAPI low-latency helper component.
- [BreakingBad (AI-Hobbyist)](https://www.ai-hobbyist.com/thread-1143-1-1.html) - Source of original models in integration pack.
