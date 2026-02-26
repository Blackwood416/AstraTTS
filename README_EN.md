# AstraTTS

<p align="center">
  <img src="images/banner.png" alt="AstraTTS Banner" width="800">
</p>

<p align="center">
  <strong>🎙️ High-Performance Cross-Platform TTS (Text-to-Speech) Engine</strong>
</p>

<p align="center">
  A high-performance speech synthesis solution based on ONNX Runtime, supporting streaming output, concurrent inference, and multi-voice management.
</p>

<p align="center">
  English | <a href="./README.md">简体中文</a>
</p>

---

## 📢 New Version v1.2.1 Released!

AstraTTS v1.2.1 brings comprehensive deployment optimizations and experience upgrades, particularly for Linux and containerized environments.

### ✨ New Features & Improvements
- 🐳 **Native Docker Support** - Includes a streamlined Dockerfile for deployment, built on `.NET 10 (Ubuntu Noble)` and a native Python environment. Extremely fast and easy zero-config startup setup.
- 🗃️ **Git LFS Resource Hosting** - `resources-minimal` (core models, dictionaries) and `tools` (model converter scripts) are now fully version-controlled via Git LFS. Simply run `git lfs pull` to seamlessly acquire all required components without manual ZIP downloads.
- 🔄 **WebUI Quick Reset** - The Web management dashboard now features a one-click reset option, making it convenient for users to initialize or restore global configurations.
- 🐧 **Linux Audio Compatibility** - The audio playback component in the CLI for Linux environments now features intelligent silent fallback handling (`pw-play` -> `paplay` -> `aplay`), improving compatibility across different distributions.
- 🚀 **Full v2ProPlus & Concurrency Support** - Continues to maintain and optimize parallel loading and concurrent synthesis processing for models based on the GPT-SoVITS V2ProPlus architecture.

---

## ✨ Features

- 🚀 **High-Performance** - Powered by ONNX Runtime and optimized for CPU instruction sets, far exceeding traditional Python-based inference.
- ⚖️ **High Concurrency** - Built-in inference pool design allows simultaneous processing of multiple synthesis requests.
- 🎵 **Streaming Synthesis** - Millisecond-level first-chunk latency with "play-while-synthesizing" support.
- 🎭 **Visual Management** - Powerful WebUI dashboard for voice bank management, model conversion, and parameter tuning.
- 🔧 **Flexible Deployment** - Ideal for both lightweight CLI use and distributed Web API services on Windows or Linux.
- 🐧 **Linux Support** - Fully compatible with Linux distributions (Ubuntu, Arch, WSL) using high-performance C# core.
- 🌐 **Mixed-Language Synthesis** - Robust support for ZH-EN and ZH-JA bilingual mixing (trilingual mix is still in development).
- 🔄 **Hot Reload** - Configuration changes take effect immediately without service interruption.

## 📦 Project Structure

- **AstraTTS.Core**: Core SDK containing hybrid G2P engines, RoBERTa/Hubert extractors, and high-performance inference modules.
- **AstraTTS.CLI**: Command-line tool. Supports low-latency WASAPI on Windows; on Linux, it pipes audio to `aplay`, `paplay`, or `pw-play`.
- **AstraTTS.Web**: Backend Web service with a comprehensive management dashboard, suitable for headless Linux server deployment.

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

### For Windows Users (`-win64.zip`):
1. **Download and Extract** the integration package.
2. **Start Web Dashboard**: Run `astra-server.exe`.
3. **Access Interface**: Open `http://localhost:5000` in your browser.
   - Use the "Converter" tab to import your SoVITS models.
   - Use the "Avatars" tab to upload reference audios and tune parameters.
4. **CLI Testing**: Run `astra-cli.exe` for local low-latency playback.

### For Linux Users (`-linux64.tar.gz`):
1. **Download and Extract**: `tar -xzvf AstraTTS-v*-linux64.tar.gz`
2. **Initialize Converter Environment**:
   - The Linux package does not embed the heavy Python runtime. You need to manually initialize the dependencies for model conversion.
   - Run `./init-env.sh` in the extracted root directory (this auto-creates a lightweight Python virtual environment under `tools/converter/.venv`).
3. **Start the Engine**:
   - Make it executable and run: `chmod +x astra-server && ./astra-server`.
   - For local CLI testing, try: `./astra-cli --text "Testing"`.

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
- **Windows**: Windows 10/11 (x64/arm64).
- **Linux**: Major distributions (Ubuntu 22.04+, Arch Linux, WSL2).
  - **Linux Dependencies**: `dotnet-runtime-10.0` is required.
  - **Linux Audio**: Install `alsa-utils` (`aplay`) or `pulseaudio` (`paplay`) for CLI playback.

### Build & Publish
- **Windows**: Run `publish.ps1`.
- **Linux**: Run `publish-linux.sh`.

### Docker Deployment (Recommended for Servers)
The project includes a multi-stage optimized Dockerfile that seamlessly integrates with Models managed by Git LFS. Deployment is native-grade and takes only a few steps:

```bash
# 1. Clone the repository (If you are downloading the source code for the first time)
git clone https://github.com/Blackwood416/AstraTTS.git
cd AstraTTS

# 2. Pull Git LFS Model Resources (CRITICAL step, otherwise the built image will lack the actual model binaries)
git lfs pull

# 3. Build the Docker image (Optimized and fast, dependencies are cached)
docker build -t astratts-server:latest .

# 4. Run the container
# It is highly recommended to mount the local resources directory with -v so you can manage your models directly from the host machine.
docker run -d --name astratts \
  -p 5000:5000 \
  -v ./resources:/app/resources \
  astratts-server:latest
```

After the container starts, access `http://localhost:5000` in your browser to view the complete Web management dashboard. Config hot-reloading, voice bank management, AND model conversion UI are all fully supported while running inside the container.

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
