# AstraTTS

<p align="center">
  <img src="images/banner.png" alt="AstraTTS Banner" width="800">
</p>

<p align="center">
  <strong>🎙️ High-Performance Cross-Platform TTS (Text-to-Speech) Engine</strong>
</p>

<p align="center">
  A high-quality speech synthesis solution based on ONNX Runtime, supporting streaming output, multi-voice management, and low-latency playback.
</p>

<p align="center">
  English | <a href="./README.md">简体中文</a>
</p>

---

## 🚀 Stable Releases (Recommended)

If you don't want to configure the build environment, you can directly download our pre-built **Portable Package**:

👉 **[Download Latest Stable Release](https://github.com/Blackwood416/AstraTTS/releases/latest)**

> [!TIP]
> The portable package comes with all necessary runtimes and the V1 default voice. Just extract and follow the [Portable Package Usage](#-portable-package-usage) below.

---

## ✨ Features

- 🚀 **High-Performance** - Powered by ONNX Runtime, deeply optimized for CPU inference.
- 🎵 **Streaming Synthesis** - Real-time audio streaming with extremely low first-chunk latency.
- 🎭 **Voice Management** - Supports custom voice banks and reference audio configurations.
- 🔧 **Flexible Deployment** - Interactive CLI tool and standard Web API service provided.
- 🌐 **Mixed Language** - Built-in Chinese-English G2P system for natural mixed-language synthesis.
- 🔄 **Hot Reload** - Dynamically reload configurations without stopping the service.

## 📦 Project Structure

- **AstraTTS.Core**: The core SDK containing inference engines, text processing (G2P/TextNorm), and audio utilities.
- **AstraTTS.CLI**: A command-line interactive tool for one-shot synthesis and real-time streaming tests.
- **AstraTTS.Web**: An ASP.NET Core based web service providing RESTful APIs and a simple documentation interface.

---

## 🔧 Engine Versions

AstraTTS supports two inference engines. **V1 is currently the recommended stable version.**

| Feature | V1 Engine (Recommended) | V2 Engine (Experimental) |
| :--- | :--- | :--- |
| **Origin** | Based on [Genie-TTS](https://github.com/High-Logic/Genie-TTS) | Based on [GPT-SoVITS-Minimal](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) |
| **Status** | ✅ Stable, Production Ready | 🚧 WIP, Experimental |
| **Acceleration** | CPU Only (Optimized) | CPU Only (GPU WIP) |
| **Speed Control** | ✅ Supported (`Speed`) | ✅ Supported (`Speed`) |
| **Sampling Params** | ❌ Not Supported | ✅ Supported (`TopK`, `Temperature`) |
| **Noise Scale** | ❌ Not Supported | ✅ Supported (`NoiseScale`) |

#### 📂 Models & Resources

Resources are mainly located in the `resources` directory:

- **V1 Models (`resources/models_v1/default/`)**: Contains `tts/`, `bert/`, `hubert/`, and `speaker_encoder.onnx`.
- **V2 Models (`resources/models_v2/default/`)**: Experimental GPT-SoVITS models.
- **Shared Resources (`resources/shared/`)**:
  - `dictionaries/cmudict.dict`: English pronunciation dictionary.
  - `dictionaries/mandarin_pinyin.dict`: Chinese Pinyin dictionary.
  - `dictionaries/opencpop-strict.txt`: Chinese G2P core dictionary.
  - `g2p/checkpoint20.npz`: English Neural G2P model.
  - `custom_dict.txt`: User-defined custom dictionary (supports hot reload).

---

## 📦 Portable Package Usage

1. **Download and Extract** the integration package.
2. **Start WebAPI**: Run `astra-server.exe`. It runs at `http://localhost:5000` by default.
3. **Local Testing**: Run `astra-cli.exe` to enter interactive mode.
4. **Custom Config**: Simply edit `config.json` for live updates.

---

## 🛠️ Model Conversion Tool (GPT-SoVITS to V1)

We have integrated the Python conversion script in the `tools/converter` directory to convert standard GPT-SoVITS `.ckpt` and `.pth` files to the ONNX format used by Astra-TTS V1:

1. **Initialize Environment** (Requires Python 3.9+):
   ```powershell
   cd tools/converter
   ./init_env.ps1
   ```
2. **Run Conversion**:
   ```powershell
   ./venv/Scripts/python.exe v1_converter.py --ckpt <GPT_CKPT_PATH> --pth <SoVITS_PTH_PATH> --shells ./templates --out ./output_dir
   ```

---

## 🚀 Developer Quick Start (Source Build)

### Prerequisites
- .NET 10.0 SDK or higher.
- Windows 10/11 (WASAPI components currently require Windows).

### 1. Configuration
Copy `config.template.json` to `config.json`. Update `ResourcesDir` if necessary.

### 2. Build & Run
```bash
# Build the solution
dotnet build

# Start CLI Interactive mode
dotnet run --project AstraTTS.CLI

# Start Web API service
dotnet run --project AstraTTS.Web
```

---

## ⚙️ Configuration Guide (`config.json`)

The following guide is based on the **V1 Engine** configuration:

```json
{
  "ResourcesDir": "resources",      // Root resource directory
  "UseEngineV2": false,             // Whether to use V2 engine (Suggested: false)
  "DefaultAvatarId": "default",     // Default voice ID
  
  "IntraOpNumThreads": 0,           // ONNX internal thread count (0 = auto)
  "InterOpNumThreads": 0,           // ONNX operator thread count
  
  "Speed": 1.0,                     // Speed (0.5 - 2.0, V1/V2 supported)
  
  "StreamingMode": true,            // Enable streaming synthesis
  "StreamingChunkSize": 22,         // Min tokens for stream trigger (V1 Only)
  
  "WasapiExclusiveMode": true,      // (CLI) WASAPI exclusive mode for playback
  
  "Avatars": [                      // Voice library
    {
      "Id": "default",
      "Name": "Default Voice",
      "References": [               // Reference audio
        {
          "Id": "normal",
          "AudioPath": "normal.wav",// WAV audio path (Rel to avatar references dir)
          "Text": "..."             // Transcription of the reference audio
        }
      ]
    }
  ]
}
```

---

## 📄 License
MIT License

## 🙏 Acknowledgments
- [Genie-TTS](https://github.com/High-Logic/Genie-TTS) - Core architecture reference for V1 engine.
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - Algorithm source for V2 engine.
- [GPT-SoVITS Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) - C# inference reference for V2.
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance cross-platform backend.
- [NAudio](https://github.com/naudio/NAudio) - .NET Audio processing.
- [BreakingBad (AI-Hobbyist)](https://www.ai-hobbyist.com/thread-1143-1-1.html) - Source of original models in integration pack.
